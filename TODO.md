TODOs:

Note: Save all screenshots under assets/screenshots.

- [ ] Build and ship the User Manual
  - [ ] Source present: [manual.typ](./manual.typ)
  - [ ] Compile output generated: [manual.pdf](./manual.pdf)
  - [ ] Install Typst (direct download): https://github.com/typst/typst/releases
  - [ ] Add typst.exe to PATH

  ```
  typst compile .\manual.typ
  ```

- [ ] Fill user-facing sections in the manual
  - [ ] Deployment & Installation: Step-by-step instructions (OS, Python version, libraries, CPU/GPU needs, environment setup). Write for non-technical users.
  - [ ] Main Features: Simple description of what the system does and benefits (avoid jargon).
  - [ ] Primary Walkthrough: Clear steps for main usage; include screenshots where noted in the manual.
  - [ ] Additional Walkthroughs: At least two scenarios (e.g., backend inference, retraining). Include screenshots.

- [ ] Capture screenshots (save images to assets/screenshots and insert where indicated)
  - [ ] Web app: launch page, upload step, prediction + confidence result.
  - [ ] Backend inference: terminal command, input path, printed result.
  - [ ] Training: dataset folders (yes/no), training log/epoch output, updated model file timestamp.

- [ ] Final checks
  - [ ] Manual compiles; PDF opens without blocking errors.
  - [ ] Links in manual and README work (commands and paths verified).
  - [ ] Title/author info and date are up to date.
