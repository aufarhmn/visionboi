const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const zmq = require('zeromq');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: "http://localhost:3000",
        methods: ["GET", "POST"],
        credentials: true
    }
});

const zmqSocket = new zmq.Subscriber();
const ZMQ_ADDRESS = 'tcp://127.0.0.1:5555';

(async () => {
    await zmqSocket.connect(ZMQ_ADDRESS);
    await zmqSocket.subscribe('');

    console.log("Waiting for data...");

    for await (const [data] of zmqSocket) {
        const portAndData = data.toString().split(':');

        if (portAndData.length === 2) {
            const port = portAndData[0];
            const [angleValue, distanceValue] = portAndData[1].split(',');

            console.log(`Received: Port=${port}, Angle=${angleValue}, Distance=${distanceValue}`);

            io.emit('message', { port, angleValue, distanceValue });
        } else {
            console.error("Invalid data received.");
        }
    }
})().catch(err => {
    console.error('Error in ZeroMQ subscriber:', err);
});

io.on('connection', (socket) => {
    console.log('New client connected');

    socket.on('disconnect', () => {
        console.log('Client disconnected');
    });
});

const PORT = 5000;
server.listen(PORT, () => {
    console.log(`Server listening on http://localhost:${PORT}`);
});
