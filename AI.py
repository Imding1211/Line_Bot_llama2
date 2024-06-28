from llama_cpp import Llama

def AI_reply(question):

    llm = Llama(
        model_path="ggml-model-q4_0.gguf",
        n_gpu_layers=100,
        verbose=False,
    )

    output = llm(
        f"### USER: {question}\n\n### ASSISTANT:",
        max_tokens=256,
        stop=["###"],
        stream=True,
    )

    reply = []
    for token in output:
        reply.append(token["choices"][0]["text"])

    return ''.join(reply)