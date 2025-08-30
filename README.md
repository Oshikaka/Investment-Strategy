# 📘 Investment-Strategy

This repository provides tools and strategies for investment analysis.  
The following guide is written for **beginners with no coding experience** — just follow step by step.

---

## 1. Install Required Software (if you don’t already have it).  

1. **Install [Visual Studio Code](https://code.visualstudio.com/)**  
   - Download and install according to your operating system (Windows / Mac / Linux).  
   - After installation, open **Visual Studio Code (VS Code)**.  

2. **Install [Anaconda](https://www.anaconda.com/download)**  
   - Anaconda is a tool to manage Python environments.  
   - After installation, check if it works:  
     - On Windows: open **Anaconda Prompt**.  
     - On Mac/Linux: open **Terminal**.  
     - Type:
       ```bash
       conda --version
       ```
       If you see a version number, Anaconda is installed correctly.  

3. **Install [Git](https://git-scm.com/downloads)** 
   - Git is required to download this project from GitHub.  


## 2. Download the Project

1. Open **VS Code**.  
2. From the top menu, click **Terminal → New Terminal** to open a terminal window.  
3. In the terminal, type:
   ```bash
   git clone https://github.com/Oshikaka/Investment-Strategy.git
   ```
   This will create a folder named Investment-Strategy in your current directory(folder).
   

## 3. Set Up the Environment

Go into the project folder and create the environment:
```bash
cd Investment-Strategy
conda env create -f environment.yml
```

This may take a while since it installs all required libraries.

## 4. Activate the Environment

Activate the environment before running the code:

```bash
conda activate invest
```


## 5. Run the Code

Navigate to the folder where your Python file is located, for example:
```bash
cd folder_you_want_to_go
python file_name.py
```

To go back to the previous folder:
```bash
cd ..
```
## 🔧 Common Issues

`command not found: conda`
→ Anaconda is not installed correctly. Reinstall and check “Add to PATH” during setup.

`ModuleNotFoundError`
→ You forgot to activate the environment. Run:
```bash
conda activate invest
```

Not sure which folder you are in?

On `Mac/Linux`:

```bash
ls
```

On `Windows`:
```bash
dir
```

## 💡 Each time you want to run the project, you need to:

```bash
cd Investment-Strategy
conda activate invest
python file_name.py  
```

