"""[클라우드 인프라] LLM 생성

- 에이전트가 쓰는 LLM 객체를 만드는 곳입니다. 다른 파일에서 ChatOpenAI(...) 를 직접 만들지 말고
  `get_llm(config)` 를 불러 쓰세요. 그래야 모델을 바꿀 때 이 파일만 고치면 됩니다.
- config["llm"]["provider"] 로 OpenAI 와 오픈소스(L4 GPU) 모델을 바꿔가며 비교 실험합니다.
  - openai: OpenAI API (.env 의 OPENAI_API_KEY 필요)
  - ollama: GCP VM 의 Ollama 서버. SSH 터널(-L 11434:localhost:11434)을 연 상태에서 사용
  - vllm: GCP VM 의 vLLM 서버(OpenAI 호환 API). SSH 터널(-L 8000:localhost:8000)을 연 상태에서 사용
"""

from langchain_core.language_models import BaseChatModel


def get_llm(config: dict) -> BaseChatModel:
    llm_cfg = config["llm"]

    if llm_cfg["provider"] == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=llm_cfg["model"], temperature=llm_cfg.get("temperature", 0))

    if llm_cfg["provider"] == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            model=llm_cfg["model"],
            temperature=llm_cfg.get("temperature", 0),
            num_ctx=llm_cfg.get("num_ctx", 8192),  # Ollama 기본값(4096)은 RFP 청크를 넣기에 작음
            # gemma4 는 기본으로 생각(thinking) 토큰을 먼저 만들어서 느리고, 컨텍스트를 다 쓰면 답이 비어 버림
            reasoning=llm_cfg.get("reasoning"),
        )

    if llm_cfg["provider"] == "vllm":
        from langchain_openai import ChatOpenAI

        extra_body = {}
        if llm_cfg.get("enable_thinking") is not None:
            # qwen 계열은 기본으로 생각(thinking) 토큰을 먼저 만듦. 채팅 템플릿 옵션으로 끔 (gemma4 reasoning=False 와 같은 목적)
            extra_body["chat_template_kwargs"] = {"enable_thinking": llm_cfg["enable_thinking"]}
        return ChatOpenAI(
            base_url=llm_cfg.get("base_url", "http://localhost:8000/v1"),
            api_key=llm_cfg.get("api_key", "EMPTY"),  # vLLM 은 키 검사를 하지 않음
            model=llm_cfg["model"],  # vLLM 이 띄운 모델 이름 (curl localhost:8000/v1/models 의 id)
            temperature=llm_cfg.get("temperature", 0),
            max_tokens=llm_cfg.get("max_tokens"),
            extra_body=extra_body or None,
        )

    raise ValueError(f"지원하지 않는 llm provider: {llm_cfg['provider']}")
