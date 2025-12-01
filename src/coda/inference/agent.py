import json
import datetime
import asyncio


class InferenceClient:
    def __init__(self, host: str = "localhost", port: int = 5123):
        self.host = host
        self.port = port

    async def start(self):
        """Start the inference server"""
        server = await asyncio.start_server(self.handle, self.host, self.port)
        print(f"Inference server listening on {self.host}:{self.port}")

        async with server:
            await server.serve_forever()

    async def handle(self, reader: asyncio.StreamReader,
                     writer: asyncio.StreamWriter):
        """Handle incoming dialogue chunks"""
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break

                message = json.loads(data.decode())

                # Simulate inference processing
                await asyncio.sleep(0.5)
                results = await self.process_chunk(message["text"])

                # Send results back
                response = {
                    "chunk_id": message["chunk_id"],
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def process_chunk(self, text: str) -> dict:
        raise NotImplementedError



class CodaToyInferenceAgent(InferenceClient):
    async def process_chunk(self, text: str) -> dict:
        return {
            "cod": "COVID-19" if "fever" in text.lower() else "cardiac arrest",
        }
