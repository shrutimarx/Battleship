#include <WiFi.h>
#include <Adafruit_NeoPixel.h>
#include <esp_task_wdt.h>
#include <Baljeet06-project-1_inferencing.h> // Edge Impulse exported model

// ── LED Grid ─────────────────────────────────────────────────────────────────
const uint8_t BOARD_SIZE = 8;
#define LED_PIN     0
#define NUM_LEDS    64
#define NUM_SQUARES 64
#define WDT_TIMEOUT 10  // seconds before watchdog reboots the board

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

#define COLOR_OFF   strip.Color(0,   0,   0  )
#define COLOR_IDLE  strip.Color(0,   5,   20 )  // ocean blue
#define COLOR_SHIP  strip.Color(0,   50,  40 )  // teal — own unhit ships
#define COLOR_HIT   strip.Color(200, 20,  0  )  // red
#define COLOR_MISS  strip.Color(0,   30,  120)  // blue
#define COLOR_SUNK  strip.Color(220, 80,  0  )  // orange

//  Wi-Fi
const char* ssid     = "Battleship";
const char* password = "seabattle";

WiFiServer server(80);
String header;

unsigned long lastActivityTime = 0;
const unsigned long IDLE_INTERVAL = 60000; // sleep after 60s idle

// tracks what the AI has fired at — synced via /ai_result
// 0=unknown, 1=miss, 2=hit
int ai_memory_board[NUM_SQUARES] = {0};
int human_memory_board[NUM_SQUARES] = {0}; // reserved for future use

// LED Helpers 

// serpentine layout: even rows right-to-left, odd rows left-to-right
// flip the condition if LEDs appear horizontally mirrored
int xyToIndex(int row, int col) {
  return (row % 2 == 0) ? row * 8 + (7 - col) : row * 8 + col;
}

void setCell(int row, int col, uint32_t color) {
  strip.setPixelColor(xyToIndex(row, col), color);
  strip.show();
}

void flashSunk(int row, int col) {
  int idx = xyToIndex(row, col);
  for (int i = 0; i < 3; i++) {
    strip.setPixelColor(idx, COLOR_OFF);  strip.show(); delay(80);
    strip.setPixelColor(idx, COLOR_SUNK); strip.show(); delay(80);
  }
}

void clearGrid() {
  for (int i = 0; i < NUM_LEDS; i++) strip.setPixelColor(i, COLOR_IDLE);
  strip.show();
}

// flash red 3x as a low-power warning, then enter deep sleep
// wakeup via PB
void goToSleep() {
  Serial.println("System idling... triggering low power warning.");
  for (int pulse = 0; pulse < 3; pulse++) {
    for (int i = 0; i < NUM_LEDS; i++) strip.setPixelColor(i, strip.Color(150, 0, 0));
    strip.show(); delay(400);
    for (int i = 0; i < NUM_LEDS; i++) strip.setPixelColor(i, COLOR_OFF);
    strip.show(); delay(200);
  }
  Serial.println("Entering Deep Sleep now.");
  esp_sleep_enable_ext1_wakeup(1ULL << 6, ESP_EXT1_WAKEUP_ANY_HIGH);
  esp_deep_sleep_start();
}

//  AI Inference (Medium mode — Edge Impulse model)

// Edge Impulse callback-- converts ai_memory_board into one-hot float input
// each cell becomes 3 floats: [unknown, miss, hit]
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
  for (size_t i = 0; i < length; i += 3) {
    int square_index = (offset + i) / 3;
    int state = ai_memory_board[square_index];
    out_ptr[i + 0] = (state == 0) ? 1.0f : 0.0f; // Unknown
    out_ptr[i + 1] = (state == 1) ? 1.0f : 0.0f; // Miss
    out_ptr[i + 2] = (state == 2) ? 1.0f : 0.0f; // Hit
  }
  return 0;
}

// run the trained model and return the best unshot cell (flat index 0-63)
int get_ai_action_medium() {
  signal_t features_signal;
  features_signal.total_length = NUM_SQUARES * 3; // 64 cells * 3 one-hot channels
  features_signal.get_data = &raw_feature_get_data;

  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);
  if (res != EI_IMPULSE_OK) {
    Serial.print("ERR: Failed to run classifier, error code: ");
    Serial.println(res);
    return 0;
  }

  float best_value = -999999.0f;
  int best_action = -1;

  // pick highest Q-value among cells not yet fired at
  for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    if (ai_memory_board[i] == 0) {
      float q_value = result.classification[i].value;
      if (q_value > best_value) {
        best_value = q_value;
        best_action = i;
      }
    }
  }

  // fallback: pick first unshot cell if model returns nothing valid
  if (best_action == -1) {
    for (int i = 0; i < NUM_SQUARES; i++) {
      if (ai_memory_board[i] == 0) return i;
    }
  }
  return best_action;
}

// Setup
void setup() {
  Serial.begin(115200);
  delay(1000);

  // testing
  esp_sleep_wakeup_cause_t wakeup_reason = esp_sleep_get_wakeup_cause();
  lastActivityTime = millis();

  // watchdog: reboot if loop() hangs for WDT_TIMEOUT seconds
  esp_task_wdt_config_t twdt_config = {
    .timeout_ms     = WDT_TIMEOUT * 1000,
    .idle_core_mask = (1 << 0), // watch core 0 where loop() runs
    .trigger_panic  = true
  };
  esp_task_wdt_init(&twdt_config);
  esp_task_wdt_add(NULL);

  strip.begin();
  strip.setBrightness(30);
  clearGrid();

  WiFi.softAP(ssid, password);
  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP: "); Serial.println(IP);
  server.begin();
}

// Loop
void loop() {
  esp_task_wdt_reset(); // feed the watchdog every iteration
  if (millis() - lastActivityTime > IDLE_INTERVAL) goToSleep();

  WiFiClient client = server.available();
  if (!client) return;

  lastActivityTime = millis(); // reset idle timer on new connection
  String currentLine = "";

  while (client.connected()) {
    if (!client.available()) continue;
    char c = client.read();
    header += c;

    if (c != '\n') {
      if (c != '\r') currentLine += c;
      continue;
    }

    if (currentLine.length() > 0) { currentLine = ""; continue; }

    // Route HTTP requests

    if (header.indexOf("GET /hit") >= 0) {
      lastActivityTime = millis();
      int row = getParam(header, "r"), col = getParam(header, "c");
      if (row >= 0 && col >= 0) setCell(row, col, COLOR_HIT);
      sendOK(client); break;

    } else if (header.indexOf("GET /miss") >= 0) {
      lastActivityTime = millis();
      int row = getParam(header, "r"), col = getParam(header, "c");
      if (row >= 0 && col >= 0) setCell(row, col, COLOR_MISS);
      sendOK(client); break;

    } else if (header.indexOf("GET /sunk") >= 0) {
      lastActivityTime = millis();
      int row = getParam(header, "r"), col = getParam(header, "c");
      if (row >= 0 && col >= 0) { flashSunk(row, col); setCell(row, col, COLOR_SUNK); }
      sendOK(client); break;

    // encoding: '0'=idle '1'=ship '2'=hit '3'=miss '4'=sunk
    } else if (header.indexOf("GET /sync_leds") >= 0) {
      lastActivityTime = millis();
      int idx = header.indexOf("state=");
      if (idx >= 0) {
        idx += 6; // move past "state="
        for (int i = 0; i < 64 && idx < (int)header.length(); i++, idx++) {
          char ch = header[idx];
          int r = i / 8, col = i % 8;
          uint32_t cColor = COLOR_IDLE;
          if      (ch == '1') cColor = COLOR_SHIP;
          else if (ch == '2') cColor = COLOR_HIT;
          else if (ch == '3') cColor = COLOR_MISS;
          else if (ch == '4') cColor = COLOR_SUNK;
          strip.setPixelColor(xyToIndex(r, col), cColor);
        }
        strip.show(); // push all 64 updates at once
      }
      sendOK(client); break;

    // difficulty passed as ?diff=0 (easy), ?diff=1 (medium), ?diff=2 (impossible mode, ai literally knows where the ships are)
    } else if (header.indexOf("GET /ai_move") >= 0) {
      lastActivityTime = millis();
      int diff = getParam(header, "diff");
      int action = 0;

      if (diff == 0) {
        // Easy: pick a random unshot cell
        int candidates[NUM_SQUARES], count = 0;
        for (int i = 0; i < NUM_SQUARES; i++) {
          if (ai_memory_board[i] == 0) candidates[count++] = i;
        }
        if (count > 0) action = candidates[random(count)];

      } else if (diff == 2) {
        int shipIdx = header.indexOf("ships=");
        if (shipIdx >= 0) {
          shipIdx += 6;
          for (int i = 0; i < NUM_SQUARES && shipIdx < (int)header.length(); i++, shipIdx++) {
            if (header[shipIdx] == '1' && ai_memory_board[i] == 0) {
              action = i;
              break;
            }
          }
        } else {
          // fallback if ships param missing
          action = get_ai_action_medium();
        }

      } else {
        // Medium (default): use the trained Edge Impulse model
        action = get_ai_action_medium();
      }

      client.println("HTTP/1.1 200 OK");
      client.println("Content-type:application/json");
      client.println("Connection: close\n");
      client.print("{\"r\":"); client.print(action / 8);
      client.print(",\"c\":"); client.print(action % 8);
      client.println("}");
      break;

    // res: 1=miss, 2=hit
    } else if (header.indexOf("GET /ai_result") >= 0) {
      lastActivityTime = millis();
      int row = getParam(header, "r"), col = getParam(header, "c"), res = getParam(header, "res");
      if (row >= 0 && col >= 0 && row < 8 && col < 8) {
        ai_memory_board[row * 8 + col] = res;
      }
      sendOK(client); break;

    } else if (header.indexOf("GET /reset") >= 0) {
      lastActivityTime = millis();
      clearGrid();
      for (int i = 0; i < NUM_SQUARES; i++) ai_memory_board[i] = 0;
      sendOK(client); break;

    // intentional hang to verify watchdog — remove before production!
    } else if (header.indexOf("GET /crash") >= 0) {
      Serial.println("TEST: Triggering intentional hang for Watchdog...");
      sendOK(client); delay(500);
      while (true);

    } else {
      client.println("HTTP/1.1 200 OK");
      client.println("Content-type:text/html");
      client.println("Connection: close\n");
      sendGamePage(client);
      break;
    }
  }
  header = "";
}

// Helpers
void sendOK(WiFiClient &client) {
  client.println("HTTP/1.1 200 OK");
  client.println("Content-type:text/plain");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Connection: close\n");
  client.println("ok");
}

// parse a single integer query param from the HTTP header
int getParam(String &h, String key) {
  String search = key + "=";
  int idx = h.indexOf(search);
  if (idx < 0) return -1;
  idx += search.length();
  String val = "";
  while (idx < (int)h.length() && isDigit(h[idx])) val += h[idx++];
  return val.length() > 0 ? val.toInt() : -1;
}

// Game Page 
void sendGamePage(WiFiClient &client) {
  client.print(F("<!DOCTYPE html><html lang='en'><head>"));
  client.print(F("<meta charset='UTF-8'>"));
  client.print(F("<meta name='viewport' content='width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no'>"));
  client.print(F("<title>Battleship vs AI</title>"));
  client.print(F("<link href='https://fonts.googleapis.com/css2?family=Pirata+One&family=Share+Tech+Mono&display=swap' rel='stylesheet'>"));
  client.print(F("<style>"));
  client.print(F("*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent;user-select:none}"));
  client.print(F(":root{--deep:#04111e;--gold:#c9a84c;--gold-l:#f0d080;--teal:#1abc9c;--red:#c0392b;--red-g:#e74c3c;--blue:#2980b9;--orange:#e67e22;--txt:#d4eaf7;--dim:#6a9ab8}"));
  client.print(F("body{background:var(--deep);color:var(--txt);font-family:'Share Tech Mono',monospace;min-height:100vh;overflow-x:hidden;background-image:radial-gradient(ellipse 80% 40% at 50% 0%,rgba(10,60,100,.5) 0%,transparent 70%),radial-gradient(ellipse 60% 30% at 50% 100%,rgba(4,17,30,.8) 0%,transparent 70%)}"));
  client.print(F("body::before{content:'';position:fixed;inset:0;background-image:repeating-linear-gradient(180deg,transparent 0px,transparent 38px,rgba(26,96,128,.06) 38px,rgba(26,96,128,.06) 40px);pointer-events:none;z-index:0}"));
  client.print(F("#app{position:relative;z-index:1;max-width:460px;margin:0 auto;padding:10px 12px 20px}"));
  client.print(F("header{text-align:center;padding:12px 0 8px;border-bottom:1px solid rgba(201,168,76,.25);margin-bottom:10px}"));
  client.print(F(".title{font-family:'Pirata One',cursive;font-size:clamp(24px,7vw,36px);color:var(--gold);letter-spacing:.08em;text-shadow:0 0 20px rgba(201,168,76,.4),0 2px 0 rgba(0,0,0,.5)}"));
  client.print(F(".title-sub{font-size:9px;letter-spacing:.35em;color:var(--dim);text-transform:uppercase;margin-top:2px}"));
  client.print(F(".banner{font-size:12px;text-align:center;padding:7px 14px;background:rgba(10,51,72,.7);border:1px solid rgba(201,168,76,.2);border-radius:4px;color:var(--txt);margin-bottom:10px;min-height:32px;display:flex;align-items:center;justify-content:center;letter-spacing:.04em}"));
  client.print(F(".banner b{color:var(--gold-l);font-weight:normal;font-family:'Pirata One',cursive;font-size:14px}"));

  // difficulty selector styles
  client.print(F(".diff-row{display:flex;gap:6px;margin-bottom:10px;justify-content:center}"));
  client.print(F(".diff-btn{flex:1;max-width:120px;padding:7px 4px;font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:.05em;border:1px solid rgba(201,168,76,.3);border-radius:3px;background:rgba(10,51,72,.7);color:var(--dim);cursor:pointer;text-transform:uppercase;transition:all .15s;text-align:center}"));
  client.print(F(".diff-btn.active-easy{border-color:#2ecc71;color:#2ecc71;background:rgba(46,204,113,.1)}"));
  client.print(F(".diff-btn.active-medium{border-color:var(--gold);color:var(--gold);background:rgba(201,168,76,.1)}"));
  client.print(F(".diff-btn.active-hard{border-color:#e74c3c;color:#e74c3c;background:rgba(231,76,60,.1)}"));

  client.print(F(".grid-wrap{display:flex;gap:3px;margin-bottom:8px}"));
  client.print(F(".col-labels{display:flex;margin-bottom:2px}"));
  client.print(F(".col-label{flex:1;text-align:center;font-size:9px;color:var(--dim)}"));
  client.print(F(".row-labels{display:flex;flex-direction:column;width:18px}"));
  client.print(F(".row-label{font-size:9px;color:var(--dim);height:0;display:flex;align-items:center;justify-content:flex-end;padding-right:3px}"));
  client.print(F(".grid{display:grid;grid-template-columns:repeat(8,1fr);gap:2px;flex:1}"));
  client.print(F(".cell{aspect-ratio:1;border-radius:2px;border:1px solid rgba(26,96,128,.4);background:rgba(7,32,48,.85);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:11px;position:relative;overflow:hidden;transition:background .15s,border-color .15s,transform .1s}"));
  client.print(F(".cell::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(26,96,128,.08) 0%,transparent 60%);pointer-events:none}"));
  client.print(F(".cell.ship-own{background:rgba(26,188,156,.18);border-color:rgba(26,188,156,.6)}.cell.ship-own::after{content:'\\25AA';color:rgba(26,188,156,.7);font-size:10px}"));
  client.print(F(".cell.preview{background:rgba(26,188,156,.12);border-color:rgba(26,188,156,.5)}.cell.preview-invalid{background:rgba(192,57,43,.15);border-color:rgba(192,57,43,.5)}"));
  client.print(F(".cell.start-sel{background:rgba(26,188,156,.2);border:2px solid var(--teal);box-shadow:inset 0 0 6px rgba(26,188,156,.3)}"));
  client.print(F(".cell.hit{background:rgba(192,57,43,.25);border-color:var(--red);cursor:default}"));
  client.print(F(".cell.hit::after{content:'';display:block;width:60%;height:60%;background:var(--red-g);clip-path:polygon(50% 0%,61% 35%,98% 35%,68% 57%,79% 91%,50% 70%,21% 91%,32% 57%,2% 35%,39% 35%);filter:drop-shadow(0 0 3px var(--red-g));animation:hitPop .3s ease-out}"));
  client.print(F("@keyframes hitPop{from{transform:scale(0);opacity:0}to{transform:scale(1);opacity:1}}"));
  client.print(F(".cell.miss{background:rgba(237,234,222,.12);border-color:rgba(237,234,222,.3);cursor:default}.cell.miss::after{content:'\\25E6';color:var(--blue);font-size:14px;opacity:.7}"));
  client.print(F(".cell.sunk{background:rgba(230,126,34,.2);border-color:var(--orange);cursor:default}.cell.sunk::after{content:'\\2715';color:var(--orange);font-size:11px;font-weight:bold;filter:drop-shadow(0 0 2px var(--orange))}"));
  client.print(F(".cell.peek{background:rgba(26,188,156,.13);border-color:rgba(26,188,156,.45)}.cell.peek::after{content:'\\25AA';color:rgba(26,188,156,.6);font-size:10px}"));
  client.print(F(".cell.target{cursor:crosshair}.cell.target:hover{background:rgba(192,57,43,.12);border-color:rgba(192,57,43,.5);transform:scale(0.93)}"));
  client.print(F("@keyframes sunkFlash{0%{filter:brightness(2.5)}100%{filter:brightness(1)}}.cell.sunk-anim{animation:sunkFlash .5s ease-out}"));
  client.print(F(".pip-row{display:flex;gap:5px;justify-content:center;margin-bottom:6px}"));
  client.print(F(".pip{width:11px;height:11px;border-radius:2px;border:1px solid rgba(201,168,76,.3);background:rgba(7,32,48,.8);transition:background .3s}"));
  client.print(F(".pip.placed{background:var(--teal);border-color:var(--teal);box-shadow:0 0 4px rgba(26,188,156,.5)}"));
  client.print(F(".action-row{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap}"));
  client.print(F(".btn{flex:1;min-width:80px;padding:9px 10px;font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:.06em;border:1px solid rgba(201,168,76,.3);border-radius:3px;background:rgba(10,51,72,.7);color:var(--gold);cursor:pointer;transition:background .15s,border-color .15s,transform .1s;text-transform:uppercase}"));
  client.print(F(".btn:hover{background:rgba(201,168,76,.12);border-color:rgba(201,168,76,.6)}.btn:active{transform:scale(.97)}"));
  client.print(F(".btn.peek-on{background:rgba(26,188,156,.15);border-color:rgba(26,188,156,.6);color:var(--teal)}"));
  client.print(F(".btn.danger{border-color:rgba(231,76,60,.3);color:#e74c3c}.btn.danger:hover{background:rgba(231,76,60,.1);border-color:rgba(231,76,60,.6)}"));
  client.print(F(".log{font-size:11px;text-align:center;color:var(--dim);min-height:18px;margin:4px 0;letter-spacing:.04em}.log.good{color:#2ecc71}.log.bad{color:#e74c3c}.log.warn{color:var(--gold)}"));
  client.print(F(".winner-box{text-align:center;font-family:'Pirata One',cursive;font-size:26px;color:var(--gold);text-shadow:0 0 20px rgba(201,168,76,.5);padding:16px;background:rgba(10,51,72,.6);border:1px solid rgba(201,168,76,.35);border-radius:4px;margin-top:10px;letter-spacing:.08em}"));
  client.print(F(".winner-sub{font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--dim);margin-top:4px;letter-spacing:.1em}"));
  client.print(F("</style></head><body>"));

  // HTML
  client.print(F("<div id='app'>"));
  client.print(F("<header><div class='title'>&#9875; Battleship</div><div class='title-sub'>Player vs AI Intelligence</div></header>"));

  // difficulty selector — only shown during placement phase
  client.print(F("<div class='diff-row' id='diff-row'>"));
  client.print(F("<div class='diff-btn active-easy' id='d0' onclick='setDiff(0)'>Easy</div>"));
  client.print(F("<div class='diff-btn' id='d1' onclick='setDiff(1)'>Medium</div>"));
  client.print(F("<div class='diff-btn' id='d2' onclick='setDiff(2)'>Impossible</div>"));
  client.print(F("</div>"));

  client.print(F("<div class='banner' id='banner'>Loading...</div>"));
  client.print(F("<div class='pip-row' id='pips'></div>"));
  client.print(F("<div class='grid-wrap'><div class='row-labels' id='rows'></div><div style='flex:1'><div class='col-labels' id='cols'></div><div class='grid' id='grid'></div></div></div>"));
  client.print(F("<div class='log' id='log'></div>"));
  client.print(F("<div class='action-row' id='actions'></div>"));
  client.print(F("<div id='winner-box'></div>"));
  client.print(F("</div>"));

  // javaScript 
  client.print(F("<script>"));
  client.print(F("var R=8,C=8,SHIPS=4;"));
  client.print(F("var COL_L=['A','B','C','D','E','F','G','H'];"));
  client.print(F("var ROW_L=['1','2','3','4','5','6','7','8'];"));

  // difficulty: 0=easy (random), 1=medium (AI model), 2=impossible (knows ship locations)
  client.print(F("var difficulty=0;"));

  // currentPlayer: 0=human, 1=AI
  client.print(F("var phase='place1';var currentPlayer=0;"));
  client.print(F("var selectStart=null,peeking=false;"));
  // defense[p]=ship positions, attack[p]=shot history
  client.print(F("var defense=[mkB(),mkB()];var attack=[mkB(),mkB()];var ships=[[],[]];"));

  client.print(F("function mkB(){return Array.from({length:R},function(){return Array.from({length:C},function(){return{ship:false,hit:false,miss:false,sunkMark:false};});});}"));
  client.print(F("function cel(r,c){return document.getElementById('g-'+r+'-'+c);}"));

  // difficulty selector- locked once battle starts
  client.print(F("function setDiff(d){"
    "if(phase!=='place1')return;"
    "difficulty=d;"
    "['d0','d1','d2'].forEach(function(id,i){"
      "var el=document.getElementById(id);"
      "el.className='diff-btn'+(i===d?(d===0?' active-easy':d===1?' active-medium':' active-hard'):'');"
    "});"
  "}"));

  client.print(F("function syncPhysicalGrid(){"
    "var s='';"
    "var showH=(phase==='place1'||currentPlayer===1||peeking);"
    "for(var r=0;r<8;r++){for(var c=0;c<8;c++){"
      "if(phase==='place1'){s+=defense[0][r][c].ship?'1':'0';}"
      "else if(showH){"
        "var a=attack[1][r][c],d=defense[0][r][c];"
        "if(a.sunkMark)s+='4';else if(a.hit)s+='2';else if(a.miss)s+='3';else if(d.ship)s+='1';else s+='0';"
      "}else{"
        "var a=attack[0][r][c];"
        "if(a.sunkMark)s+='4';else if(a.hit)s+='2';else if(a.miss)s+='3';else s+='0';"
      "}"
    "}}"
    "fetch('/sync_leds?state='+s).catch(function(){});"
  "}"));

  client.print(F("function buildGrid(){"
    "var grid=document.getElementById('grid'),colsEl=document.getElementById('cols'),rowsEl=document.getElementById('rows');"
    "colsEl.innerHTML='';rowsEl.innerHTML='';grid.innerHTML='';"
    "COL_L.forEach(function(l){var d=document.createElement('div');d.className='col-label';d.textContent=l;colsEl.appendChild(d);});"
    "for(var r=0;r<R;r++){"
      "var rl=document.createElement('div');rl.className='row-label';rl.textContent=ROW_L[r];rowsEl.appendChild(rl);"
      "for(var c=0;c<C;c++){"
        "var cell=document.createElement('div');cell.className='cell';cell.id='g-'+r+'-'+c;"
        "(function(rr,cc){"
          "cell.addEventListener('click',function(){onCellClick(rr,cc);});"
          "cell.addEventListener('mouseenter',function(){onHover(rr,cc);});"
          "cell.addEventListener('mouseleave',function(){clearPreview();});"
        "})(r,c);"
        "grid.appendChild(cell);"
      "}"
    "}"
    "requestAnimationFrame(function(){"
      "var first=grid.querySelector('.cell');if(!first)return;"
      "rowsEl.querySelectorAll('.row-label').forEach(function(el){el.style.height=first.getBoundingClientRect().height+'px';});"
    "});"
  "}"));

  client.print(F("function updatePips(){"
    "var el=document.getElementById('pips');el.innerHTML='';"
    "if(phase!=='place1')return;"
    "for(var i=0;i<SHIPS;i++){var pip=document.createElement('div');pip.className='pip'+(i<ships[0].length?' placed':'');el.appendChild(pip);}"
  "}"));

  client.print(F("function renderGrid(){"
    "for(var r=0;r<R;r++){for(var c=0;c<C;c++){"
      "var el=cel(r,c);el.className='cell';"
      "if(phase==='place1'){"
        "if(defense[0][r][c].ship){el.classList.add('ship-own');}"
      "}else if(phase==='battle'){"
        "var atk=attack[currentPlayer][r][c];"
        "if(atk.sunkMark){el.classList.add('sunk');}"
        "else if(atk.hit){el.classList.add('hit');}"
        "else if(atk.miss){el.classList.add('miss');}"
        "else{"
          "if(peeking&&defense[currentPlayer][r][c].ship){el.classList.add('peek');}"
          "else if(!atk.hit&&!atk.miss&&!atk.sunkMark){el.classList.add('target');}"
        "}"
      "}"
    "}}"
  "}"));

  client.print(F("function renderAll(){renderGrid();updatePips();updateBanner();updateActions();}"));

  client.print(F("function onCellClick(r,c){"
    "if(phase==='place1'){doPlace(0,r,c);}"
    "else if(phase==='battle')doFire(r,c);"
  "}"));
  client.print(F("function onHover(r,c){if(phase==='place1'&&selectStart)showPreview(0,r,c);}"));

  client.print(F("function doPlace(p,r,c){"
    "if(ships[0].length>=SHIPS)return;"
    "if(!selectStart){"
      "if(defense[0][r][c].ship){setLog('A ship is already there.',badCls);return;}"
      "selectStart={r:r,c:c};cel(r,c).className='cell start-sel';"
      "setLog('Start chosen at '+COL_L[c]+ROW_L[r]+'. Now click the end cell.');"
    "}else{"
      "var cells=lineOf(selectStart,{r:r,c:c});"
      "if(!cells){setLog('Must be a straight line.',badCls);selectStart=null;renderGrid();return;}"
      "if(cells.length<1||cells.length>4){setLog('Ship must be 1-4 cells.',badCls);selectStart=null;renderGrid();clearPreview();return;}"
      "if(cells.some(function(rc){return defense[0][rc[0]][rc[1]].ship;})){setLog('Overlaps an existing ship.',badCls);selectStart=null;renderGrid();clearPreview();return;}"
      "cells.forEach(function(rc){defense[0][rc[0]][rc[1]].ship=true;});"
      "ships[0].push({cells:cells,sunk:false});"
      "selectStart=null;clearPreview();renderGrid();updatePips();"
      "syncPhysicalGrid();"
      "var rem=SHIPS-ships[0].length;"
      "if(rem>0){setLog('Ship placed! '+rem+' more to go.');}"
      "else{"
        // all ships placed — lock difficulty, auto-place AI, start battle
        "document.getElementById('diff-row').style.opacity='0.4';"
        "document.getElementById('diff-row').style.pointerEvents='none';"
        "autoPlaceAI();"
        "phase='battle';currentPlayer=0;peeking=false;"
        "notifyReset();renderAll();syncPhysicalGrid();"
        "var diffNames=['Easy','Medium','Impossible'];"
        "setLog('Setup complete! Difficulty: '+diffNames[difficulty]+'. You attack first.');"
      "}"
    "}"
  "}"));

  // randomly place AI ships with fixed sizes [4,3,3,2]
  client.print(F("function autoPlaceAI(){"
    "var lens=[4,3,3,2];"
    "for(var i=0;i<SHIPS;i++){"
      "var placed=false;"
      "while(!placed){"
        "var r=Math.floor(Math.random()*R),c=Math.floor(Math.random()*C),hz=Math.random()>0.5;"
        "var cells=[];"
        "for(var l=0;l<lens[i];l++){if(hz)cells.push([r,c+l]);else cells.push([r+l,c]);}"
        "var valid=cells.every(function(rc){return rc[0]<R&&rc[1]<C&&!defense[1][rc[0]][rc[1]].ship;});"
        "if(valid){"
          "cells.forEach(function(rc){defense[1][rc[0]][rc[1]].ship=true;});"
          "ships[1].push({cells:cells,sunk:false});"
          "placed=true;"
        "}"
      "}"
    "}"
  "}"));

  client.print(F("function lineOf(a,b){"
    "if(a.r===b.r){var cc=[];for(var c=Math.min(a.c,b.c);c<=Math.max(a.c,b.c);c++)cc.push([a.r,c]);return cc;}"
    "if(a.c===b.c){var rc=[];for(var r=Math.min(a.r,b.r);r<=Math.max(a.r,b.r);r++)rc.push([r,a.c]);return rc;}"
    "return null;"
  "}"));

  client.print(F("function showPreview(p,r,c){"
    "clearPreview();"
    "var cells=lineOf(selectStart,{r:r,c:c});if(!cells)return;"
    "var valid=cells.length>=1&&cells.length<=4&&!cells.some(function(rc){return defense[0][rc[0]][rc[1]].ship;});"
    "cells.forEach(function(rc){cel(rc[0],rc[1]).classList.add(valid?'preview':'preview-invalid');});"
  "}"));
  client.print(F("function clearPreview(){document.querySelectorAll('.preview,.preview-invalid').forEach(function(el){el.classList.remove('preview','preview-invalid');});}"));

  client.print(F("function doFire(r,c){"
    "if(currentPlayer!==0)return;" // block taps during AI turn
    "var atk=attack[0][r][c];"
    "if(atk.hit||atk.miss||atk.sunkMark){setLog('Already fired there.',badCls);return;}"
    "if(peeking)peeking=false;"
    "var def=defense[1][r][c];"
    "if(def.ship){"
      "atk.hit=true;"
      "var ship=null;"
      "for(var i=0;i<ships[1].length;i++){"
        "var sh=ships[1][i];"
        "if(!sh.sunk&&sh.cells.some(function(rc){return rc[0]===r&&rc[1]===c;})){ship=sh;break;}"
      "}"
      "var sunkNow=false;"
      "if(ship){"
        "var allHit=ship.cells.every(function(rc){return attack[0][rc[0]][rc[1]].hit;});"
        "if(allHit){"
          "ship.sunk=true;sunkNow=true;"
          "ship.cells.forEach(function(rc){"
            "attack[0][rc[0]][rc[1]].sunkMark=true;attack[0][rc[0]][rc[1]].hit=false;"
            "notifyLED('sunk',rc[0],rc[1]);"
          "});"
          "setLog('Enemy ship sunk! You fire again.','good');"
        "}else{"
          "setLog('Hit! You fire again.','good');"
          "notifyLED('hit',r,c);"
        "}"
      "}"
      "renderAll();"
      "if(sunkNow){"
        "ship.cells.forEach(function(rc){"
          "var el=cel(rc[0],rc[1]);el.classList.add('sunk-anim');"
          "setTimeout(function(){el.classList.remove('sunk-anim');},2000);"
        "});"
      "}"
      "if(ships[1].every(function(sh){return sh.sunk;})){endGame(0);}"
    "}else{"
      "atk.miss=true;"
      "notifyLED('miss',r,c);"
      "setLog('Miss! AI is taking its turn...','warn');"
      "currentPlayer=1;"
      "renderAll();"
      "setTimeout(syncPhysicalGrid,1500);"
      "setTimeout(doAITurn,2000);"
    "}"
  "}"));

  // build the ship bitmask string for impossible mode
  // 64-char string where '1' = human ship cell
  client.print(F("function buildShipMask(){"
    "var s='';"
    "for(var r=0;r<8;r++){for(var c=0;c<8;c++){"
      "s+=defense[0][r][c].ship?'1':'0';"
    "}}"
    "return s;"
  "}"));

  client.print(F("function doAITurn(){"
    "if(phase!=='battle'||currentPlayer!==1)return;"
    // build URL — impossible mode sends the ship bitmask so ESP32 can target directly
    "var url='/ai_move?diff='+difficulty;"
    "if(difficulty===2){url+='&ships='+buildShipMask();}"
    "fetch(url).then(function(r){return r.json();}).then(function(data){"
      "var r=data.r,c=data.c;"
      "var atk=attack[1][r][c];"
      "var def=defense[0][r][c];"
      "var resState=1;" // default miss; set to 2 if hit
      "var sunkNow=false,ship=null;"
      "if(def.ship){"
        "atk.hit=true;resState=2;"
        "for(var i=0;i<ships[0].length;i++){"
          "var sh=ships[0][i];"
          "if(!sh.sunk&&sh.cells.some(function(rc){return rc[0]===r&&rc[1]===c;})){ship=sh;break;}"
        "}"
        "if(ship){"
          "var allHit=ship.cells.every(function(rc){return attack[1][rc[0]][rc[1]].hit;});"
          "if(allHit){"
            "ship.sunk=true;sunkNow=true;"
            "ship.cells.forEach(function(rc){attack[1][rc[0]][rc[1]].sunkMark=true;attack[1][rc[0]][rc[1]].hit=false;});"
            "setLog('AI sunk your ship! AI fires again.','bad');"
          "}else{"
            "setLog('AI hit your ship! AI fires again.','bad');"
          "}"
        "}"
      "}else{"
        "atk.miss=true;"
        "setLog('AI missed. Your turn!','good');"
      "}"
      // update LEDs with AI's shot
      "if(sunkNow){"
        "ship.cells.forEach(function(rc){notifyLED('sunk',rc[0],rc[1]);});"
      "}else if(def.ship){"
        "notifyLED('hit',r,c);"
      "}else{"
        "notifyLED('miss',r,c);"
      "}"
      // report result back so ai_memory_board stays in sync
      "fetch('/ai_result?r='+r+'&c='+c+'&res='+resState).then(function(){"
        "renderAll();"
        "if(sunkNow&&ship){"
          "ship.cells.forEach(function(rc){"
            "var el=cel(rc[0],rc[1]);el.classList.add('sunk-anim');"
            "setTimeout(function(){el.classList.remove('sunk-anim');},2000);"
          "});"
        "}"
        "if(ships[0].every(function(sh){return sh.sunk;})){endGame(1);}"
        "else if(resState===2){setTimeout(doAITurn,2000);}" // AI hit — goes again
        "else{"
          "currentPlayer=0;"
          "setTimeout(syncPhysicalGrid,1500);"
          "setTimeout(renderAll,2000);"
        "}"
      "});"
    "});"
  "}"));

  client.print(F("function endGame(winner){"
    "phase='over';renderAll();"
    "var msg=winner===0?'You are Victorious':'AI is Victorious';"
    "var sub=winner===0?'All enemy vessels sunk \\u2014 the seas are yours':'Your fleet was destroyed \\u2014 the AI wins';"
    "document.getElementById('winner-box').innerHTML='<div class=\"winner-box\">'+msg+'<div class=\"winner-sub\">'+sub+'</div></div>';"
  "}"));

  client.print(F("var badCls='bad';"));
  client.print(F("function setLog(msg,cls){var el=document.getElementById('log');el.textContent=msg;el.className='log '+(cls||'');}"));

  client.print(F("function updateBanner(){"
    "var el=document.getElementById('banner');"
    "if(phase==='place1'){var pl=ships[0].length;el.innerHTML='<b>Deploy Fleet</b> &mdash; place ship '+(pl+1)+' of '+SHIPS;}"
    "else if(phase==='battle'){el.innerHTML=(currentPlayer===0)?'<b>Your Turn</b> &mdash; fire on enemy waters':'<b>Enemy Turn</b> &mdash; AI is targeting...';}"
    "else{el.innerHTML='Battle complete';}"
  "}"));

  client.print(F("function updateActions(){"
    "var row=document.getElementById('actions');row.innerHTML='';"
    "if(phase==='place1'&&selectStart){"
      "row.appendChild(mkBtn('Cancel','',function(){selectStart=null;clearPreview();renderGrid();setLog('Selection cleared.');updateActions();}));"
    "}"
    "if(phase==='battle'&&currentPlayer===0){"
      "row.appendChild(mkBtn(peeking?'Hide My Ships':'Peek at My Ships',peeking?'peek-on':'',function(){peeking=!peeking;renderGrid();syncPhysicalGrid();}));"
    "}"
    "row.appendChild(mkBtn('New Game','danger',function(){initGame();}));"
  "}"));

  client.print(F("function mkBtn(label,cls,fn){var b=document.createElement('button');b.className='btn '+(cls||'');b.textContent=label;b.onclick=fn;return b;}"));
  client.print(F("function notifyLED(type,r,c){fetch('/'+type+'?r='+r+'&c='+c).catch(function(){});}"));
  client.print(F("function notifyReset(){fetch('/reset').catch(function(){});}"));

  client.print(F("function initGame(){"
    "phase='place1';currentPlayer=0;selectStart=null;peeking=false;"
    "defense=[mkB(),mkB()];attack=[mkB(),mkB()];ships=[[],[]];"
    "document.getElementById('winner-box').innerHTML='';"
    "document.getElementById('diff-row').style.opacity='1';"
    "document.getElementById('diff-row').style.pointerEvents='auto';"
    "setDiff(difficulty);" // re-apply current difficulty highlight
    "setLog('Click start then end cell to place a ship (1-4 cells, straight).');"
    "buildGrid();renderAll();notifyReset();"
  "}"));

  client.print(F("initGame();"));
  client.print(F("</script></body></html>"));
}
