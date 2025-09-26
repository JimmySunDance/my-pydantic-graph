from __future__ import annotations as __annotations

from dataclasses import dataclass, field
from pydantic import BaseModel, EmailStr

from pydantic_ai import Agent, format_as_xml
from pydantic_ai.messages import ModelMessage
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from load_models import OLLAMA_MODEL

@dataclass
class User:
    name: str
    email: EmailStr
    interests: list[str]

@dataclass
class State:
    user: User
    write_agent_msg: list[ModelMessage] = field(default_factory=list)

@dataclass
class Email(BaseModel):
    subject: str
    body: str



class EmailRequiresWrite(BaseModel):
    feedback: str

class EmailOk(BaseModel):
    pass


email_writer_agent = Agent(
    model=OLLAMA_MODEL, 
    output_type=Email, 
    system_prompt="Write a welcome email for the people who subscribe to my tech blog.",
)

feedback_agent = Agent[None, EmailRequiresWrite | EmailOk](
    model=OLLAMA_MODEL, 
    output_type=EmailRequiresWrite | EmailOk,
    system_prompt=(
        "Review the email and provide feedback, email must reference the users specific interests."
    )
)


XML_PROMPT = """
<example>
    <name>John Doe</name>
    <email>john.doe@example.com</email>
</example>
"""


@dataclass
class WriteEmail(BaseNode[State]):
    email_feedback: str | None = None

    async def run(self, ctx: GraphRunContext[State]) -> Feedback:
        print(f"--- WriteEmail triggered ---")
        
        if self.email_feedback:
            prompt = (
                f"Rewrite the email for user:\n",
                f"{format_as_xml(ctx.state.user)}\n",
                f"Feedback: {self.email_feedback}"
            )
        else:
            user_xml = XML_PROMPT
            prompt = (
                f"Write a welcome email for user:\n",
                f"{user_xml}"
            )
        
        result = await email_writer_agent.run(
            user_prompt=prompt, 
            message_history=ctx.state.write_agent_msg
        )

        ctx.state.write_agent_msg += result.all_messages()
        return Feedback(result.output)


@dataclass
class Feedback(BaseNode[State, None, Email]):
    email: Email

    async def run(self, ctx: GraphRunContext[State]) -> WriteEmail | End[Email]:
        print("--- Feedback triggered ---") 

        prompt = format_as_xml({"user": ctx.state.user, "email":self.email})
        result = await feedback_agent.run(prompt)

        if isinstance(result.output, EmailRequiresWrite):
            return WriteEmail(email_feedback=result.output.feedback)
        else:
            return End(self.email)
        

if __name__ == "__main__":

    user_1 = User(
        name="Jay", email="jay@example.com",
        interests=["AI Agents", "Photography", "Hiking"]
    )

    state = State(user=user_1)

    print("*"*40)
    print(state)
    print("*"*40)

    feedback_graph = Graph(nodes=[WriteEmail, Feedback])
    email = feedback_graph.run_sync(WriteEmail(), state=state)

    email = Email.model_validate(email.output)
    
    print("\n", "--- "*15, "Final Message", " ---"*15, "\n")
    print("Subject:", email.subject)
    print("Body:", email.body)


    print("--- saving graph as .png ---")
    feedback_graph.mermaid_save("email_graph.png", infer_name=True)
    print("Done!")