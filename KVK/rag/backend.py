from flask import Flask,render_template,request,jsonify
from return_context import return_context
import ollama
with open(r'C:\Users\kumar\Desktop\KVKDEV\Aroha\KVK\rag\apikey.txt','r+') as mf:
    apikey=mf.readline()
app=Flask(__name__)
user_message_arr=[]
bot_answer_arr=[]
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/Chatbot',methods=['POST'])
def chatbot():
    if request.method == 'POST':
        userip = request.form.get('message')
        user_message_arr.append(userip)
        context = return_context(userip)
        prompt=f"""You are an expert assistant.
                Answer the question ONLY using the provided context.
                Context:
                {context}
                Question:
                {userip}
                Instructions:
                - Use ONLY information from the context.
                - Only include data that matches the user's query (e.g., state, location, etc.).
                - Ignore any entries that do not match the requested state or location.
                - Do NOT combine data from different states.
                - If no matching data is found in the context, say:
                    "No relevant data found for the specified query."
                - Do NOT make assumptions or add extra information.
                Answer:
            """
        messages = [
            {
                'role': 'user',
                'content':prompt,
            },
        ]
        response=ollama.Client(host='https://ollama.com',headers={'Authorization': 'Bearer ' + apikey}).chat(
            model='gpt-oss:120b',
            messages=messages
        )
        print(context)
        bot_answer=response['message']['content']
        return jsonify(str(bot_answer).strip('/n').strip('/n/n'))
app.run(debug=True,port=2000)