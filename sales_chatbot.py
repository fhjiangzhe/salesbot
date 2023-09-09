# 导入所需的库
import gradio as gr
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import logging
import argparse

# 定义 SalesBot 类
class SalesBot:
    def __init__(self, vector_store_dir="benz_sales_data"):
        # 初始化 SalesBot
        self.initialize(vector_store_dir)

    def initialize(self, vector_store_dir):
        # 加载本地的奔驰向量存储
        db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings())
        # 加载聊天模型（这里使用的是 GPT-3.5 Turbo）
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        # 创建一个基于检索的 QA 系统
        self.sales_bot = RetrievalQA.from_chain_type(llm,
                                                     retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                               search_kwargs={"score_threshold": 0.8}))
        self.sales_bot.return_source_documents = True

    def chat(self, message, history):
        try:
            # 记录接收到的消息和聊天历史
            logging.info(f"Received message: {message}")
            logging.info(f"Chat history: {history}")

            # 拼接 message 和 history
            full_query = f"{history} {message}"

            # 使用 QA 系统生成答案
            ans = self.sales_bot({"query": full_query})

            # 如果找到了相关的源文档，返回答案
            if ans["source_documents"]:
                logging.info(f"Result: {ans['result']}")
                logging.info(f"Source documents: {ans['source_documents']}")
                return ans["result"].replace('[销售回答]', '').strip()
            elif ans["result"]:
                logging.info(f"Result: {ans['result']}")
                return ans["result"].replace('[销售回答]', '').strip()
            else:
                return "这个问题我要问问领导"
        except Exception as e:
            # 如果出现错误，记录错误并返回一个错误消息
            logging.error(f"An error occurred: {e}")
            return "抱歉，出现了一个错误。"

# 定义一个函数，用于启动 Gradio 界面
def launch_gradio(sales_bot):
    demo = gr.ChatInterface(
        fn=sales_bot.chat,  # 设置聊天函数
        title="奔驰车销售",  # 设置标题
        chatbot=gr.Chatbot(height=600),  # 设置聊天界面的高度
    )
    # 启动 Gradio 界面
    demo.launch(share=True, server_name="0.0.0.0")

# 主程序入口
if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Sales Bot")
    parser.add_argument("--vector_store_dir", default="benz_sales_data", help="Directory for vector store")
    args = parser.parse_args()

    # 创建 SalesBot 实例
    sales_bot = SalesBot(vector_store_dir=args.vector_store_dir)
    # 启动 Gradio 界面
    launch_gradio(sales_bot)
