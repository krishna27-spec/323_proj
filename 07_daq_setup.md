# 07 · Live acquisition setup: NI drivers and checks

Code: `realtime.py` (live or replay), `scope.py` + `scope.html` (live plots in the browser), `train_realtime.py` (trains the models in `weights_rt/`), `predict_test.py`, `causal.py`.
Python environment: `.venv` (packages in `requirements.txt`, including `nidaqmx==1.6.0`).

## Hardware path

Accelerometer coupler outputs (X, Y, Z) go by BNC to ai0, ai1, ai2 of an **NI 9775** in slot 4 of a **cDAQ-9174** chassis, which connects to this PC by USB. Photos: `test/IMG_20261015_*.jpg`.

| Item | Value |
|---|---|
| Chassis | cDAQ-9174 (4 slots, USB), name `cDAQ1`, serial 0x223C13A |
| Module | NI 9775, slot 4, name `cDAQ1Mod4`, serial 0x1F647E6 |
| NI 9775 | 4 BNC inputs, ±10 V, 14-bit digitizer, up to 20 MS/s per channel, single-ended (shield = chassis ground) |
| Rate used | 20 kHz, set exactly (20 MS/s / 1000), the same as the `.lvm` recordings |
| Slot 3 | NI 9222, name `cDAQ1Mod3`, serial 0x2254B30: 4 ch ±10 V, 16-bit, simultaneous, channel-to-channel isolated, screw terminals (+/− per channel). Not used by `realtime.py` yet |
| Slots 1–2 | empty |

- `realtime.py` now defaults to `--device cDAQ1Mod4 --channels ai0:2`. The channel order must match the axis order in the training recordings, which is not documented yet.
- The grey module in the photos is the NI 9222, now in slot 3 (checked 2026-10-16).
- The 9775 has no IEPE excitation, so the accelerometer must go through its coupler, as it does now.

## The `apps/` folder

`apps/NILinux2026Q3DeviceDrivers.zip` is the NI Linux device-driver bundle, 2026 Q3, unpacked in `apps/NILinux2026Q3DeviceDrivers/`. It holds one `.deb` or `.rpm` per distribution, each in a `2026Q3` and a `stream` variant.

- This PC runs Ubuntu 22.04, so the matching file is `ni-ubuntu2204-drivers-2026Q3.deb`.
- That file is only a small package (`ni-software-2026-jammy`) that enables NI's apt repository. The drivers themselves come from that repository.
- The `stream` variants follow NI's rolling feed. Use `2026Q3` to stay on a fixed release.

## Installing (only if the checks below fail)

```
sudo apt install ./apps/NILinux2026Q3DeviceDrivers/ni-ubuntu2204-drivers-2026Q3.deb
sudo apt update
sudo apt install ni-daqmx
sudo reboot
```

The kernel modules are built by DKMS, so the kernel headers must be installed (`sudo apt install linux-headers-$(uname -r)`).

## Checking that the drivers are installed

1. Packages: `dpkg -l | grep -E "ni-daqmx|libnidaqmx|ni-kal"` should list `ii` lines.
2. Kernel modules: `lsmod | grep -E "nikal|nipalk"` should show them loaded. `dkms status | grep 26.5.0` shows (module names like `ni1045tr`, not `ni-...`) whether they were built for the running kernel.
3. Python binding: `.venv/bin/python -c "from nidaqmx.system import System as S; s=S.local(); print(s.driver_version); print([d.name for d in s.devices])"` prints the driver version, then the list of devices.
4. Hardware seen: `nilsdev` (or `nilsdev --verbose`) lists the attached chassis and modules. Empty output means no hardware is visible. `lsusb | grep -i national` shows a USB chassis at the bus level.

## State on this PC (checked 2026-10-15)

- Installed: NI-DAQmx 26.5.0 (`ni-daqmx`, `libnidaqmx`, `ni-kal`, `ni-pal-dkms` and the other DKMS modules) and the repository package `ni-software-2026-jammy`.
- Kernel modules `nikal`, `nipalk`, `nimru2k` and related ones are loaded on kernel 6.8.0-138-generic.
- `.venv` imports `nidaqmx` and reports driver 26.5.0. The system `python3` has no `nidaqmx`, so run the scripts with `.venv/bin/python`.
- With the chassis on USB, `lsusb` shows `3923:74a5 National Instruments Corp. cDAQ 9174` and `nilsdev` lists `cDAQ1` and `cDAQ1Mod4 (NI 9775)`.
- Continuous 3-channel acquisition at 20 kHz works. With nothing plugged into the BNCs, the noise is about 2.5 mV std per channel.

## Running live

```
.venv/bin/python realtime.py --rpm 20000 --log results/live.csv
```

1. Start with the spindle running and the tool in the air. The first 2 s are calibration, and every level and the model input are scaled by them.
2. Each 25.6 ms hop prints the rule and the three models (Level only, 1D CNN, CNN + GRU), each averaged over its 3 seeds. The labels are no cutting, stable or chatter. `--quiet` prints only changes.
3. Stop with Ctrl+C. The CSV is written line by line, so it survives a crash.

Timing: all 9 networks take about 11 ms per hop on the CPU against a 25.6 ms budget, and the rule takes 0.2 ms.

## Live plots in the browser: `scope.py`

```
.venv/bin/python scope.py                        # plots only
.venv/bin/python scope.py --rpm 20000 --models   # plus the chatter rule and the three models
.venv/bin/python scope.py --replay test/Acce_20k_3_r1.lvm --rpm 20000   # no hardware: loops a recording at the true rate
```

Then open http://localhost:8050 (`--port` changes it, and `?span=10` in the URL sets the starting span). The page shows:

- the X, Y, Z time signals, with spans from 50 ms to 20 s, DC removal, and auto or fixed Y scale;
- the spectrum (Welch PSD, dB re 1 V²/Hz), with dashed lines at the spindle harmonics when `--rpm` is given;
- a waterfall of about the last 10 s;
- RMS, peak-to-peak, DC and peak frequency per axis.

With `--rpm` or `--models`, it also shows the current label and a coloured label strip under the signals. These run in their own thread, so the acquisition never waits for them. The rule calibrates on the first 2 s, as in `realtime.py`, and the **Recalibrate rule** button repeats that.

**Stop acquisition** stops the sampling and closes the DAQ task, which frees the module for other programs. **Start acquisition** opens it again, and the rule then calibrates again. `--idle` starts the scope in the stopped state. **Pause view** (or the space bar) only freezes the picture, while acquisition keeps running. **Record** writes the raw volts to `recordings/scope_<date>_<time>.lvm`, in the same layout as the lab files (time, x, y, z, tab separated), so the file can be passed to `--replay` here or in `realtime.py`. Only one program can use the module at a time, so stop `scope.py` before running `realtime.py` live, and the other way round.

### Pulling the USB cable (or losing chassis power) while the scope runs

Pulling the cable is safe for the hardware. The cDAQ-9174 is USB hot-pluggable, and it takes its power from its own supply, not from the USB, so the modules stay powered. The scope only reads analog inputs and drives no outputs, so a lost connection cannot leave anything switched on.

What the scope does when it happens:

- The read fails within 2 s at most, and the DAQ task is closed.
- The page shows a red banner, "… is not visible: the USB cable is out or the chassis has no power". The server keeps running and retries every 2 s. Once the cable is back, sampling restarts by itself, and the banner then gives the time of the loss and of the reconnection.
- A running recording is closed at the last good sample, so a file never joins data from before and after a gap. Press Record again to start a new file.
- The rule calibrates again after the reconnect, so keep the tool out of the cut for 2 s.
- The time axis does not count the gap: the sample clock continues from where it stopped.

## Known issue: a start that gives no samples

On this chassis and module, some task starts never deliver a sample, and the first read fails with `-200284` ("samples requested have not yet been acquired") even with a 30 s timeout. It happens on the start right after a run was killed without cleanup (SIGTERM, a closed terminal). Neither `reset_device()` on the module nor on the chassis prevents it, but the next start always works. `daq_source` therefore retries a start whose first block does not arrive within 2 s, up to 5 times, and prints `DAQ start N gave no samples in 2 s, restarting the task`. One retry costs about 2 s. This was tested over 6 killed runs in a row, and all of them streamed.

## The website: Chatter Monitor (`docs/`)

A static website (`docs/index.html`, `docs/style.css`, `docs/app.js`) that can be published with GitHub Pages and also runs from `scope.py`. One page shows:

- the outputs: process state (0 air cutting, 1 stable, 2 transition, 3 chatter), RMS of the real-time channel, dominant frequency;
- all accelerometers stacked (add or remove channels, 1 to 4), and the selected real-time channel enlarged on the right;
- the FFT, the STFT waterfall and the top 10 frequencies (with magnitude) side by side;
- the state over the last 30 s.

Every input is in the **Inputs** sheet: number of channels, real-time channel, name, input (ai0 to ai3) and scaling factor (mV/g) per channel, source, sampling frequency, window length (256 to 4096), window function, overlap, frequency range, waterfall history, dynamic range, units (V or g), the two time spans, Y scale and DC removal. The common ones also sit on the cards. The page remembers them in the browser.

RMS, FFT, peaks and waterfall are computed in the page from the raw samples. The process state comes from the CNN + GRU model, checked against the entropy rule: while the tool cuts, the state is *transition* when the rule says "uncertain" or names the other one of stable / chatter.

**Sources.**
- *Demo*: plays the three test runs from `docs/data/` (made by `export_site_data.py`: the cut plus 4 s before and 3 s after, int16, with the rule and CNN + GRU labels). `?run=60k_test&t=9` opens a run at a given second.
- *Live*: polls `scope.py`. Open `http://localhost:8050/app/` on the DAQ PC (Live is then the default), or choose Live on the published page; the browser may ask to allow access to the local network. `scope.py` now acquires `ai0:3` by default. The rule and the models still see only ai0 to ai2 (X, Y, Z), and only at 20 kHz. A sampling frequency change from the page closes the DAQ task and starts again with an empty buffer (`POST /config?fs=`).

**Publishing on GitHub Pages.** Push the repository (with `docs/`) to GitHub, then Settings › Pages › Build and deployment › Deploy from a branch › `main`, folder `/docs`. The site appears at `https://<user>.github.io/<repo>/`. On a free account the repository must be public.
