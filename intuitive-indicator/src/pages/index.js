import Image from "next/image";
import io from "socket.io-client";
import Car from "@/assets/car.png";
import { Inter } from "next/font/google";
import { useState, useEffect } from "react";

const inter = Inter({ subsets: ["latin"] });

// X -> March height divided by Image height
// Y -> Minimum thresshold in mm divided by container width
// INITIAL_POSITION -> Initial position 
const SCALING_FACTOR_X = 11.45;
const SCALING_FACTOR_Y = parseInt(1000 / 200);
const INITIAL_POSITION = parseInt(1890 / SCALING_FACTOR_X);

export default function Home() {
  const [dotPositions, setDotPositions] = useState([]);

  useEffect(() => {
    const socket = io("http://localhost:5000");

    socket.on("message", (data) => {
      console.log("Received message:", data);

      const newPosition = getDotPosition(
        data.port,
        parseInt(data.angleValue),
        parseInt(data.distanceValue)
      );

      setDotPositions((prevPositions) => {
        const updatedPositions = [...prevPositions, newPosition];
        return updatedPositions.slice(-5);
      });
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const getDotPosition = (port, angle, distance) => {
    const angleInRadians = (angle * Math.PI) / 180;

    let x = Math.cos(angleInRadians) * distance;
    let y = Math.sin(angleInRadians) * distance;

    x = x / SCALING_FACTOR_X;
    y = y / SCALING_FACTOR_Y;

    x = INITIAL_POSITION + x;

    return { x, y };
  };

  return (
    <main
      className={`flex min-h-screen flex-col items-center justify-start p-12 ${inter.className}`}
    >
      <div className="z-10 max-w-5xl w-full h-full items-center justify-between font-mono text-sm lg:flex">
        VisionBOI Intuitive Indicator - Capstone BO1
      </div>
      <div className="relative flex flex-row justify-center w-full h-full pt-[10%] pb-[10%]">
        {/* LEFT DIV */}
        <div className="w-[200px]"></div>

        <div>
          <Image src={Car} alt="Car" width={150} height={150} />
        </div>

        {/* RIGHT DIV */}
        <div className="w-[200px] relative">
          {dotPositions.map((position, index) => (
            <div
              key={index}
              style={{
                position: "absolute",
                top: `${position.x}px`,
                left: `${position.y}px`,
                width: "15px",
                height: "15px",
                backgroundColor: "red",
                borderRadius: "50%"
              }}
            />
          ))}
        </div>
      </div>
    </main>
  );
}
