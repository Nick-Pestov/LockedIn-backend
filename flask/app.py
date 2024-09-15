from flask import Flask, jsonify, request, render_template_string
import cohere_chat as cc
import predictor as pr

app = Flask(__name__)

# A simple in-memory structure to store tasks
tasks = []

@app.route('/', methods=['GET'])
def home():
    # Display existing tasks and a form to add a new task
    html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Todo List</title>
</head>
<body>
    <h1>Todo List</h1>
    <form action="/add" method="POST">
        <input type="text" name="task" placeholder="Enter a new task">
        <input type="submit" value="Add Task">
    </form>
    <ul>
        {% for task in tasks %}
        <li>{{ task }} <a href="/delete/{{ loop.index0 }}">x</a></li>
        {% endfor %}
    </ul>
</body>
</html>
'''
    return render_template_string(html, tasks=tasks)

@app.route('/add', methods=['POST'])
def add_task():
    # Add a new task from the form data
    task = request.form.get('task')
    if task:
        tasks.append(task)
    return home()

@app.route('/delete/<int:index>', methods=['GET'])
def delete_task(index):
    # Delete a task based on its index
    if index < len(tasks):
        tasks.pop(index)
    return home()

@app.route("/respond", methods=['GET'])
def cohere_respond():
    message = request.args.get('message')
    textbook_content = request.args.get('textbook_content')
    response = cc.respond(message, textbook_content)
    
    return jsonify({"response": response})

@app.route("/get-background", methods=['GET'])
def get_background():
    pdf_content = request.args.get('pdf_content')
    background_label = pr.predict_long_text(pdf_content).lower()
    return jsonify({"background_label": background_label})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
