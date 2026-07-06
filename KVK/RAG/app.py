from flask import Flask,render_template,request,jsonify,Response
from return_content import return_context
from google.genai import Client
import os
clt=Client(api_key=os.getenv('GOOGLE_API_KEY'))
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('chatbot.html')
@app.route('/Chatbot',methods=['POST'])
def chatbot():
    if request.method == 'POST':
        userip = request.form.get('message')
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
        res=clt.models.generate_content(model='gemini-3.1-flash-lite',contents=prompt)
        print(context)
        return res.text
app.run(debug=True,port=2000)