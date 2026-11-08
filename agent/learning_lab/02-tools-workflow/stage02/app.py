from .assistant import AfterSaleAssistant

if __name__ == "__main__":
    assistant = AfterSaleAssistant("U-100")
    for question in ("我的 O-1001 到哪了？", "O-1002 的售后怎么样？", "订单怎么还没到？"):
        print("用户:", question)
        print("助手:", assistant.answer(question))
