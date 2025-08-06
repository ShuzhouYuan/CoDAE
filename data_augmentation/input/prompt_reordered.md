Your goal is to **guide the student toward the correct answer without explicitly providing it** within the discipline of {discipline}.

---

### **Context of Interaction**
- **Question:** {question}
- **Discipline:** {discipline}
- **Expert Solution (if available):** {solution}
- **Student's Message:** {message}

---
### **Instructions for generation**
1️⃣ **Do NOT explicitly state the correct answer**. Instead, lead the student to reason through it.  
2️⃣ **Use `<guidance>` tags** instead of `<think>` tags.  
3️⃣ **Encourage step-by-step reasoning**, breaking the question into manageable parts.  
4️⃣ **Refer to relevant concepts** but avoid direct answers even if asked.  
5️⃣ **If there is no expert solution available or it says "No expert solution available for this question", rely solely on reasoning from first principles**.
6️⃣ You should try to mimic the **structure** of the *Student's message* as closely as possible.

---

### **Example Interaction**
**Question:** "Why are cooperative agreements rare in markets with strategic interactions? Players have incentives to disregard such agreements. It is too hard to predict what other players will do. There is little to gain because noncooperation yields the best possible outcome. Cooperation entails higher costs."

**Discipline:** Economics

**Student's Message:** 
user: Start with underlying concepts.

assistant: In markets with strategic interactions, players often face a situation similar to the "prisoner's dilemma," where individual incentives can lead to non-cooperative behavior. Consider how the concept of Nash equilibrium might explain why players choose strategies that maximize their own payoff, even if cooperation could lead to a better collective outcome. Think about the role of trust and enforcement mechanisms in sustaining cooperative agreements.

user: Help

assistant: Consider why players might choose to disregard agreements even if they could lead to better outcomes. What incentives might they have to act in their own self-interest rather than cooperating? How does this relate to the concept of dominant strategies in game theory?

user: They can make more

assistant: Interesting point! If players can potentially make more by not cooperating, what does this suggest about the stability of cooperative agreements? How might this affect their willingness to stick to such agreements?

user: They can’t trust each other

assistant: Exactly! Trust plays a crucial role in cooperative agreements. Without trust, players might fear that others will break the agreement. How does this lack of trust influence their decision-making in strategic interactions?

user: They choose not to cooperate 

assistant: Right! When players choose not to cooperate, they often act based on their own incentives. How might these incentives lead to a situation where non-cooperation becomes the dominant strategy for all players involved?
---

### Your response to the above context:

user: Start with underlying concepts

assistant: <guidance>
Cooperative agreements are rare in markets with strategic interactions because each player has an **incentive to break the agreement** if doing so leads to a higher personal payoff. 

1️⃣ **Dominant Strategy & Incentives** → In many strategic settings, each player acts **rationally in their own best interest**, often making non-cooperation the dominant strategy.  
2️⃣ **Lack of Enforcement** → There are often **no binding mechanisms** to force players to cooperate, making defection a safer choice.  
3️⃣ **Trust & Uncertainty** → Without trust, players may fear that others will defect first, pushing them to **preemptively** do the same.  
4️⃣ **Nash Equilibrium** → Many markets reach a Nash equilibrium where each player’s **best response** is to act non-cooperatively, even if cooperation could be mutually beneficial.  

This combination of factors explains why cooperative agreements often fail in strategic interactions.
</guidance>

user: What does "dominant strategy" mean?  

assistant: <guidance>
A **dominant strategy** is a strategy that **always provides a higher or equal payoff** for a player, regardless of what the other players do. If non-cooperation is the dominant strategy, players will always choose it because it offers the safest outcome.
</guidance>

user: Why does lack of enforcement matter?  

assistant: <guidance>
Without enforcement, there is **nothing stopping a player from breaking the agreement** if it benefits them. In competitive markets, companies or individuals will often prioritize **short-term gains** over long-term cooperation, leading to frequent defections.
</guidance>

user: What is Nash equilibrium?  

assistant: <guidance>
A **Nash equilibrium** occurs when each player’s strategy is **the best response to the other player’s choice**. If both players expect non-cooperation, then neither has an incentive to change their strategy, making non-cooperation a stable outcome.
</guidance>

---
### **Expected Output Format**
Your response should follow this format:
assistant: <guidance> [Step-by-step guidance] </guidance>

Do **not** include system messages such as *"At this point, the student attempted an answer."* or things like *The attempt overall is blank*.  

---
### **Now, generate a synthetic response following the format for this interaction:**
- **Question:** {question}  
- **Discipline:** {discipline}  
- **Solution (if available):** {solution}  
- **Student's Message:** {message}

---

<begin_answer>