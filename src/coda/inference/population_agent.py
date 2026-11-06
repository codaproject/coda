"""Inference agent passing each case to the agent of its population."""

from coda.inference.agent import InferenceAgent


class PopulationAgent(InferenceAgent):
    """Infer with the agent of the case profile's population.

    The dialogue history is kept here and shared with whichever agent infers,
    so a case can change population at any point of the interview.
    """

    def __init__(self, agents, default):
        super().__init__()
        self.agents = agents
        self.default = default

    @property
    def population(self):
        """Return the population of the current case."""
        profile = self.metadata.profile
        return profile.population if profile and profile.population \
            else self.default

    async def infer(self, chunk_id, text, annotations):
        agent = self.agents[self.population]
        agent.dialogue_history = self.dialogue_history
        agent.all_text = self.all_text
        agent.metadata = self.metadata
        agent.clinical_notes = self.clinical_notes
        return await agent.infer(chunk_id, text, annotations)

    def reset(self):
        super().reset()
        for agent in self.agents.values():
            agent.reset()

    def create_session_agent(self):
        return PopulationAgent(
            {population: agent.create_session_agent()
             for population, agent in self.agents.items()}, self.default)
