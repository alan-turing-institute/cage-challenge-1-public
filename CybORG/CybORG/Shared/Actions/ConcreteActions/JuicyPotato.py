from ipaddress import IPv4Address

from CybORG.Shared import Observation
from CybORG.Shared.Actions.ConcreteActions.EscalateAction import EscalateAction
from CybORG.Shared.Enums import OperatingSystemPatch, OperatingSystemType, OperatingSystemDistribution
from CybORG.Simulator.Host import Host
from CybORG.Simulator.Process import Process
from CybORG.Simulator.State import State


class JuicyPotato(EscalateAction):
    def __init__(self, session: int, agent: str, target_session: int):
        super().__init__(session, agent, target_session)

    def sim_execute(self, state: State) -> Observation:
        return self.sim_escalate(state, "SYSTEM")

    def test_exploit_works(self, target_host: Host):
        # the exact patches and OS distributions are described here:
        # TODO: improve checks for success
        return target_host.os_type == OperatingSystemType.WINDOWS

