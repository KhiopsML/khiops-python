REM Echo all output
@echo on

REM Build the Khiops Python package
"%PYTHON%" -m pip install . --no-deps --ignore-installed --no-cache-dir --no-build-isolation -vvv

if errorlevel 1 exit 1
