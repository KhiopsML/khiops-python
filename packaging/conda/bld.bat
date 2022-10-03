REM Echo all output
@echo on
robocopy "%CWD%"\khiops_bin\src .\src /e
robocopy "%CWD%"\khiops_bin\test .\test /e
mkdir .\packaging
robocopy "%CWD%"\khiops_bin\packaging\common .\packaging\common /e
if errorlevel 8 exit 1
copy /y "%CWD%"\khiops_bin\CMakeLists.txt .
copy /y "%CWD%"\khiops_bin\CMakePresets.json .
copy /y "%CWD%"\khiops_bin\LICENSE .
copy /y "%CWD%"\khiops_bin\packaging\install.cmake .\packaging\
copy /y "%CWD%"\khiops_bin\packaging\packaging.cmake .\packaging\

% REM We use ninja for generator because VS does not work with conda build
cmake --preset windows-msvc-release -G "Ninja" -DBUILD_JARS=OFF -DTESTING=OFF
cmake --build --preset windows-msvc-release --parallel --target MODL --target MODL_Coclustering

REM Copy the MODL binaries to the anaconda PREFIX path
mkdir %PREFIX%\bin
copy build\windows-msvc-release\bin\MODL.exe %PREFIX%\bin
copy build\windows-msvc-release\bin\MODL_Coclustering.exe %PREFIX%\bin

"%PYTHON%" -m pip install . --no-deps --ignore-installed --no-cache-dir --no-build-isolation -vvv

if errorlevel 1 exit 1
