REM Echo all output
@echo on

REM Clone Khiops sources
git clone https://github.com/khiopsml/khiops.git khiops_bin
cd .\khiops_bin\
git checkout "%KHIOPS_REVISION%"
cd ..

REM Copy relevant Khiops files to current directory
robocopy .\khiops_bin\src .\src /e
robocopy .\khiops_bin\test .\test /e
mkdir .\packaging
robocopy .\khiops_bin\packaging\common .\packaging\common /e
if errorlevel 8 exit 1
copy /y .\khiops_bin\CMakeLists.txt .
copy /y .\khiops_bin\CMakePresets.json .
copy /y .\khiops_bin\LICENSE .
copy /y .\khiops_bin\packaging\install.cmake .\packaging\
copy /y .\khiops_bin\packaging\packaging.cmake .\packaging\

REM Build the Khiops binaries
cmake --preset windows-msvc-release -DBUILD_JARS=OFF -DTESTING=OFF
cmake --build --preset windows-msvc-release --parallel --target MODL MODL_Coclustering

REM Copy the MODL binaries to the Conda PREFIX path
mkdir %PREFIX%\bin
copy build\windows-msvc-release\bin\MODL.exe %PREFIX%\bin
copy build\windows-msvc-release\bin\MODL_Coclustering.exe %PREFIX%\bin

REM Build the Khiops Python package
"%PYTHON%" -m pip install . --no-deps --ignore-installed --no-cache-dir --no-build-isolation -vvv

if errorlevel 1 exit 1
