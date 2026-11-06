import asyncio

from coda.inference.agent import InferenceAgent
from coda.inference.population_agent import PopulationAgent


class EchoAgent(InferenceAgent):
    def __init__(self, name):
        super().__init__()
        self.name = name

    async def infer(self, chunk_id, text, annotations):
        return {"causes": {}, "reasoning": f"{self.name}: {self.all_text}"}

    def create_session_agent(self):
        return EchoAgent(self.name)


def test_population_agent_switches():
    agent = PopulationAgent({"child": EchoAgent("child"),
                             "adult": EchoAgent("adult")}, "child")
    agent = agent.create_session_agent()
    result = asyncio.run(agent.process_chunk("c0", "fever", []))
    assert result["reasoning"] == "child:  fever"
    result = asyncio.run(agent.process_chunk(
        "c1", "cough", [], metadata={"profile": {"population": "adult"}}))
    assert result["reasoning"] == "adult:  fever cough"
    agent.reset()
    assert agent.agents["adult"].all_text == ""
