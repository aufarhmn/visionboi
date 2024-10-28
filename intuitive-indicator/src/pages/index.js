import Image from "next/image";
import { Inter } from "next/font/google";
import Car from "@/assets/car.png";
import { useState } from "react";

const inter = Inter({ subsets: ["latin"] });

export default function Home() {
  // IMG WIDTH: 150, HEIGHT: 330 
  // NISSAN MARCH: 3780 MM. SCALE FACTOR: 11.45
  // VALUE JARAK DARI DEPAN REAL / 11.45
  // PX AND MM
  // Y AXIS STILL NOT CALIBRATED
  const SCALING_FACTOR_X = 11.45;

  const [lidarData, setLidarData] = useState({
    port: "COM4",
    angle: 90,
    distance: 1000 
  });

  // POSITION STILL NOT CALIBRATED, USING IDEAL POINT FROM MARCH HEIGHT
  const getDotPosition = (port, angle, distance) => {
    let initialPosition = 0;
    if (port === "COM4") {
      initialPosition = 82.5;
    } else if (port === "COM10") {
      initialPosition = 247.5;
    }

    const angleInRadians = (angle * Math.PI) / 180;

    let x = (Math.cos(angleInRadians) * distance);
    let y = (Math.sin(angleInRadians) * distance);

    x = x / SCALING_FACTOR_X;
    y = y / SCALING_FACTOR_X;

    x = initialPosition + x;

    return { x, y };
  };

  const dotPosition = getDotPosition(
    lidarData.port,
    lidarData.angle,
    lidarData.distance
  );

  console.log(dotPosition);

  return (
    <main
      className={`flex min-h-screen flex-col items-center justify-start p-12 ${inter.className}`}
    >
      <div className="z-10 max-w-5xl w-full h-full items-center justify-between font-mono text-sm lg:flex">
        VisionBOI Intuitive Indicator - Capstone BO1
      </div>
      <div className="relative flex flex-row justify-center w-full h-full pt-[10%] pb-[10%]">
        <div>
          <Image src={Car} alt="Car" width={150} height={150} />
        </div>

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
    </main>
  );
}
