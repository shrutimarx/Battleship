# ⚓ Smart Battleship
### ECE 4180 Final Project - Georgia Tech
**Ridwan Siddique & Shruti Marx | Spring 2026**

---

## Overview

Smart Battleship is an embedded implementation of the classic Battleship board game built on the **ESP32-C6** microcontroller. Players interact through a custom web interface hosted directly on the ESP32, while an **8×8 WS2812B RGB LED matrix** displays the result of each shot in real time. No app installation or external server is required; the ESP32 acts as its own Wi-Fi access point and web server.

---

## Demo Photo

![IMG_0500](https://github.gatech.edu/user-attachments/assets/4f5c9ec4-28aa-4499-98f0-48d32af77b7d)

---

## How It Works

1. Connect phone to the ESP32's Wi-Fi network (`Battleship`)
2. Navigate to `192.168.4.1` in any browser
3. Place **4 ships** (length 1–4 cells) on the 8×8 grid
5. Alternate firing, a **hit gives an extra turn**, a miss passes control
6. The LED matrix mirrors the active player's attack board in real time
7. When all of a player's ships are sunk, the game ends

---

## Hardware

| Component | Purpose |
|---|---|
| ESP32-C6 | Main processor — hosts Wi-Fi AP, web server, game logic, LED control |
| 8×8 Flex LED Matrix (5V) | Displays hit/miss/sunk state in real time |
| Breadboard + Wires | Power distribution and signal routing |
| Push Button | Routed to wake up from deep sleep mode |

---

## Circuit Diagram

<img width="783" height="502" alt="image" src="https://github.gatech.edu/user-attachments/assets/5c82cb40-db39-4b39-85b3-1914ba1a0751" />

---

## Features

### 🟢 Digital Output — NeoPixel LED Matrix
The WS2812B matrix is driven over a single GPIO using the NeoPixel protocol. A serpentine index mapping function translates `(row, col)` coordinates to physical LED indices. The matrix updates on every hit, miss, sunk, and turn change.

### 📶 Wi-Fi — Soft Access Point + Web Server
The ESP32 runs in SoftAP mode broadcasting a network called `Battleship`. A `WiFiServer` on port 80 serves the full game UI as a single self-contained HTML page. HTTP GET requests from the browser (`/hit`, `/miss`, `/sunk`, `/reset`) trigger LED updates on the matrix.

### 💤 Sleep Mode
After 20 seconds of inactivity (5 minutes in production), LEDs fade out and the ESP32 enters light sleep via `esp_light_sleep_start()`. The device wakes on a HIGH signal to GPIO 15 using `gpio_wakeup_enable()`. Game state and Wi-Fi are preserved across sleep cycles.

### 🐕 Watchdog Timer
A hardware timer initialized, acts as a watchdog. It is fed by `timerWrite(wdt, 0)` on every `loop()` iteration and inside the client handler. If the system hangs beyond the timeout, an ISR sets a flag and `loop()` calls `esp_restart()` for a clean reboot.

---

## Libraries

| Library | Use | Source |
|---|---|---|
| Adafruit NeoPixel | LED matrix control | [github.com/adafruit/Adafruit_NeoPixel](https://github.com/adafruit/Adafruit_NeoPixel) |
| WiFi.h | SoftAP + TCP server | Espressif Arduino Core (built-in) |
| esp_task_wdt.h / driver/gpio.h | watchdog fed via esp_task_wdt_reset | Espressif Arduino Core (built-in) |
| Baljeet06-project-1_inferencing.h / AI inferencing | Edge Impulse | Engineer Created |

No modifications were made to any external library.

---

## Challenges

**Serving HTML over raw `WiFiServer`**
Arduino's `WiFiServer` has no async or chunked transfer support, so the entire HTML page had to be split into many small `client.print(F("..."))` calls with strings stored in flash via the `F()` macro to avoid exhausting RAM. This was pretty tedious but kept dependencies minimal and matched our lab coursework pattern.

**Challenges with the AI Model**
Memory constraints on the ESP-32 after extensive front-end code forced us to reduce the size of the model. We reduced and refactored some of they layers to allow for higher performance with little additional memory and we made sure to quantize the weights and input and output tensors as int8 data types instead of float32 to massively decrease memory footprint.

**Watchdog ISR**
Our initial implementation called `esp_restart()` directly from the watchdog ISR, which caused a core dump. `esp_restart()` is not interrupt-safe; it needs to shut down peripherals cleanly. The fix was to set a `volatile` flag inside the ISR and perform the restart from normal `loop()` context.

---

## Proposal Changes

**ML Model**
We decided to implement a machine learning model to play against the player so there is an intelligent opponent for single player games.

---

## Related Work

| Project | How Ours Differs |
|---|---|
| Hasbro Electronic Battleship | No physical pegs or bulky enclosure; it uses RGB LEDs and a phone UI |
| Online Battleship | Runs fully offline on embedded hardware, no internet or accounts needed |
| Standard Arduino LED projects | Full game logic, web server, sleep, and WDT all integrated on one chip |

---

## Future Improvements

- **Non-volatile high scores** using the ESP32 `Preferences` library to persist fastest-win records across power cycles
- **Larger grid** by chaining a second LED matrix for a 16×8 or 16×16 board
- **PWM Buzzer** General sound effects

