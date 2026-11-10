"""[클라우드 인프라] LLM 생성

- 에이전트가 쓰는 LLM 객체를 만드는 곳입니다. 다른 파일에서 ChatOpenAI(...) 를 직접 만들지 말고
  `get_llm(config)` 를 불러 쓰세요. 그래야 모델을 바꿀 때 이 파일만 고치면 됩니다.
- config["llm"]["provider"] 로 OpenAI 와 오픈소스(L4 GPU) 모델을 바꿔가며 비교 실험합니다.
"""

from langchain_core.language_models import BaseChatModel


def get_llm(config: dict) -> BaseChatModel:
    llm_cfg = config["llm"]

    if llm_cfg["provider"] == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=llm_cfg["model"], temperature=llm_cfg.get("temperature", 0))

    # TODO: 오픈소스 모델 실험을 시작하면 여기에 추가
    #   예) vLLM 등으로 띄운 OpenAI 호환 서버를 쓰면 ChatOpenAI(base_url=..., model=...) 로 동일하게 사용 가능
    raise ValueError(f"지원하지 않는 llm provider: {llm_cfg['provider']}")
