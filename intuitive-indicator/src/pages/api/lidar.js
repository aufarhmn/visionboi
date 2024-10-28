const zmq = require("zeromq");

let socket;

async function setupZMQ() {
  if (!socket) {
    socket = new zmq.Subscriber();
    try {
      socket.connect("tcp://127.0.0.1:5555");
      socket.subscribe("");
      console.log("Connected to ZeroMQ publisher");
    } catch (error) {
      console.error("Failed to connect to ZeroMQ:", error);
      throw error;
    }
  }
  return socket;
}

export default async function handler(req, res) {
  try {
    const socket = await setupZMQ();

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();

    console.log("Listening for ZeroMQ messages...");
    
    (async () => {
      for await (const [msg] of socket) {
        const data = msg.toString();
        console.log("[API] Received from ZMQ:", data);
        res.write(`data: ${data}\n\n`);
      }
    })();

    req.on("close", () => {
      console.log("Client disconnected");
      res.end();
    });
  } catch (error) {
    console.error("Error in /api/lidar:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}
