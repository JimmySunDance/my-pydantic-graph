from dataclasses import dataclass
from pydantic_graph import GraphRunContext, BaseNode, Graph, End


@dataclass
class Node_A(BaseNode[int]):
    track_number: int

    async def run(self, ctx: GraphRunContext) -> BaseNode:
        print(f"Calling Node A")
        return Node_B(self.track_number) # type: ignore

@dataclass
class Node_B(BaseNode[int]):
    track_number: int

    async def run(self, ctx: GraphRunContext) -> BaseNode | End:
        print(f"Calling Node B")
        if self.track_number == 1:
            return Node_C(self.track_number) # type: ignore
        elif self.track_number < 1:
            return End(f"Value out of scope.")  # type: ignore
        else:
            self.track_number -= 1
            print("Tracked number:", self.track_number)
            return Node_B(self.track_number)  # type: ignore

@dataclass
class Node_C(BaseNode[int]):
    track_number: int

    async def run(self, ctx: GraphRunContext) -> End:
        print("Calling Node C")
        return End(f"Value to be returned at Node C: {self.track_number}")  # type: ignore
    

if __name__ == "__main__":
    graph = Graph(nodes=[Node_A, Node_B, Node_C])

    result = graph.run_sync(start_node=Node_A(track_number=-1), state=-1)

    print("*"*40)
    print("History: ")
    print(result)

    print("-"*40)
    print("Result:", result)

    