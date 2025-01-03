import Image from "next/image";
import io from "socket.io-client";
import Car from "@/assets/car.png";
import { Inter } from "next/font/google";
import { useState, useEffect } from "react";

const inter = Inter({ subsets: ["latin"] });

// ADJUST THIS
const SCALING_FACTOR_X = 11.45;
const SCALING_FACTOR_Y = 1000 / 200;

export default function Home() {
  const [lidarData, setLidarData] = useState({
    port: "COM4",
    angle: 90,
    distance: 1000
  });

  useEffect(() => {
    const socket = io("http://localhost:5000");

    socket.on("message", (data) => {
      console.log("Received message:", data);
      setLidarData({
        port: data.port,
        angle: parseInt(data.angleValue),
        distance: parseInt(data.distanceValue)
      });
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  // POSITION STILL NOT CALIBRATED, USING IDEAL POINT FROM MARCH HEIGHT
  const getDotPosition = (port, angle, distance) => {
    let initialPosition = 0;
    if (port === "COM4") {
      initialPosition = 262;
    } else if (port === "COM10") {
      initialPosition = 247.5;
    }

    const angleInRadians = (angle * Math.PI) / 180;

    let x = Math.cos(angleInRadians) * distance;
    let y = Math.sin(angleInRadians) * distance;

    x = x / SCALING_FACTOR_X;
    y = y / SCALING_FACTOR_Y;

    x = initialPosition + x;

    return { x, y };
  };

  const dotPosition = getDotPosition(
    lidarData.port,
    lidarData.angle,
    lidarData.distance
  );

  return (
    <main
      className={`flex min-h-screen flex-col items-center justify-start p-12 ${inter.className}`}
    >
      <div className="z-10 max-w-5xl w-full h-full items-center justify-between font-mono text-sm lg:flex">
        VisionBOI Intuitive Indicator - Capstone BO1
      </div>
      <div className="relative flex flex-row justify-center w-full h-full pt-[10%] pb-[10%]">
        {/* LEFT DIV FOR NEXT DEV */}
        <div className="w-[200px]"></div>

        <div>
          <Image src={Car} alt="Car" width={150} height={150} />
        </div>

        {/* RIGHT DIV */}
        <div className="w-[200px]">
          <div
            style={{
              marginTop: `${dotPosition.x}px`,
              marginLeft: `${dotPosition.y}px`,
              width: "20px",
              height: "20px",
              backgroundColor: "red",
              borderRadius: "50%"
            }}
          />
        </div>
      </div>
    </main>
  );
}
