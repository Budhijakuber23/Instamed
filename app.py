from flask import Flask, render_template, request, redirect, session
import gemini_src
import pickle
from PIL import Image
import os

from model import labels
app = Flask(__name__)
app.secret_key = "seckey"
app.permanent_session_lifetime = 300

# initialization of gemini
gemini = gemini_src.Gemini()


def process_image(image_path):
    """
    Process the uploaded image and predict the [organ, issue].
    """
    try:
        image = Image.open(image_path)
        image_array = image.convert("RGB")  # Convert to RGB if needed

        prediction = labels(image_path)
        
        # prediction = model.predict([image_array])  # Use appropriate preprocessing
        return prediction[0], prediction[1]  # Assuming the output is [organ, issue]
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


@app.route('/', methods=['GET', 'POST'])
def main():
    session.permanent = False
    session_data = session.get('output_dict', {})
    context = session.get('context', '')

    if request.method == 'POST':
        if 'image' in request.files:
            # Handle image upload
            uploaded_file = request.files['image']
            if uploaded_file.filename != '':
                # Save the uploaded image temporarily
                image_path = os.path.join("uploads", uploaded_file.filename)
                uploaded_file.save(image_path)

                # Process the image and get the context
                result = process_image(image_path)
                if result:
                    organ, issue = result
                    context = f"The patient has an issue with {organ}. The specific issue is {issue}."
                    session['context'] = context
                    session['output_dict'] = {}  # Clear previous chat history
                else:
                    context = "Failed to analyze the uploaded image."
                    session['context'] = context

                return redirect('/')
        
        elif 'ask' in request.form:
            # Handle user query
            task_content = request.form['ask']
            try:
                prompt = f"{context}\nUser: {task_content}\nAI:"
                output = gemini.generate(prompt)
                session_data[task_content] = output
                session['output_dict'] = session_data
                return redirect('/')
            except Exception as e:
                return f"There was an issue connecting to Gemini: {e}"

    return render_template('index.html', output=session_data, context=context)


@app.route('/clear_session', methods=['GET', 'POST'])
def clear_session():
    session.pop('output_dict', None)
    session.pop('context', None)
    session.clear()
    return 'Session cleared'


if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(host='0.0.0.0', port=5000)