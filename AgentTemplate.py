class AgentTemplate:
    def __init__(self, agent_id, C_sense, C_act, C_soft, r=None):
        self.id = agent_id
        self.C_sense = set(C_sense)
        self.C_act = set(C_act)
        self.C_soft = set(C_soft)
        self.r = r  # 资源，可选

    def covers(self, C_sense, C_act, C_soft) -> bool:
        return (C_sense.issubset(self.C_sense) and
                C_act.issubset(self.C_act) and
                C_soft.issubset(self.C_soft))

    def total_capability_size(self):
        return len(self.C_sense | self.C_act | self.C_soft)