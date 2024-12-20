import os.path

from openai import OpenAI
import traceback

def generate(query):
    try:
        # OpenAI 클라이언트 초기화
        client = OpenAI(api_key='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')

        # 모델 이름 수정: gpt-4-turbo
        model = "gpt-4o"

        # ChatGPT API 요청
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ]
        )

        # 응답 메시지 내용 추출
        content = response.choices[0].message.content
        return content

    except Exception as e:
        # 예외 발생 시 에러 메시지 반환
        return (0, traceback.format_exc())

print("실행")

with open(os.path.join(os.path.dirname(__file__), 'text.txt'), "r", encoding="utf-8") as file:  # utf-8 인코딩으로 읽기
    content = file.read()

answer = generate(content)

with open(os.path.join(os.path.dirname(__file__), 'text2.txt'), "w", encoding="utf-8") as file:
    file.write(answer)
print("\r완료")