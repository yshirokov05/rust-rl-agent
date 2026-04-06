@echo off
echo Renaming old venv...
ren venv venv_old

echo Creating new Python 3.12 venv...
"C:\Users\Yuri\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv

echo Installing requirements...
call .\venv\Scripts\python -m pip install -r requirements.txt wandb pynvml
call .\venv\Scripts\python -m pip install stable-baselines3[extra] wandb[sb3]

echo Uninstalling torch...
call .\venv\Scripts\python -m pip uninstall -y torch torchvision torchaudio

echo Installing CUDA Torch...
call .\venv\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Done!
