import Image from "next/image";
import { Inter } from "next/font/google";
import Car from "@/assets/car.png";

const inter = Inter({ subsets: ["latin"] });

export default function Home() {
  return (
    <main
      className={`flex min-h-screen flex-col items-center justify-start p-12 ${inter.className}`}
    >
      <div className="z-10 max-w-5xl w-full h-full items-center justify-between font-mono text-sm lg:flex">
        VisionBOI Intuitive Indicator - Capstone BO1
      </div>
      <div className="flex flex-row items-center justify-center w-full h-full pt-[10%] pb-[10%]">
        <div>
          <Image 
            src={Car} 
            alt="Car" 
            width={150}
            height={150}
          />
        </div>
      </div>
    </main>
  );
}
