@echo off
echo Starting VisionBOI...

echo Installing python packages...
pip install -r requirements.txt

for /f "tokens=1,* delims==" %%A in (config.env) do (
    if "%%A"=="PORT_LIDAR_1" set "PORT_LIDAR_1=%%B"
    if "%%A"=="CAMERA_ID_1" set "CAMERA_ID_1=%%B"
    if "%%A"=="BAUDRATE" set "BAUDRATE=%%B"
)

echo Loaded PORT_LIDAR_1=%PORT_LIDAR_1%
echo Loaded CAMERA_ID_1=%CAMERA_ID_1%
echo Loaded BAUDRATE=%BAUDRATE%

pushd serial-comm
start /b python pub.py %PORT_LIDAR_1% --baudrate %BAUDRATE% > NUL 2>&1
popd

echo Running helper service...
pushd helper
call npm install
start /b npm run dev > NUL 2>&1
popd

echo Running intuitive indicator service...
pushd intuitive-indicator
call npm install
start /b npm run dev > NUL 2>&1
popd

echo Running real-time video stitching service...
pushd real-time-video\testing-vid-stitching
start /b python dev-rec.py %CAMERA_ID_1% > NUL 2>&1
popd

echo All services have been started in the background!
echo Keep this window open to keep the services running.
pause
