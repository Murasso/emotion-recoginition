# emotion-recoginition
顔の表情分析ツールです。
This tool facilitates facial expression analysis using Python 3.8 or 3.9.

**Note**: Make sure you have Python 3.8 or 3.9 installed.

1. Create a virtual environment:
    ```terminal
    python -m venv EmotionRecoginition
    cd EmotionRecoginiton
    ./Script/activate
    ```

2. Clone the repository:
    ```termanal
    git clone git@github.com:Murasso/emotion-recoginition.git
    ```

3. Install required libraries:
    ```termanal
    cd emotion-recoginition/application
    pip install -r requirements.txt
    ```

4. In the application directory, run the tool:
    ```terminal
    python kivyapp_ver2.py
    ```

## Required Libraries:
- absl-py==1.2.0
- altgraph==0.17.2
- astunparse==1.6.3
- backcall==0.2.0
- Bashutils==0.0.4
- ... (and many more, as listed in your provided requirements.txt)

**Notes**:
- Ensure you have the correct Python version for the virtual environment.
- The tool uses Kivy for the graphical user interface and OpenCV for facial recognition.
- Various machine learning-related libraries are listed, including TensorFlow, scikit-learn, PyTorch, etc.
- Be cautious with `git clone` using SSH, as it requires SSH keys.
- Make sure the specified library versions are compatible.
- if you press capture on the app, the image taken from your webcam gets stored in application/image folder. Make sure you have image folder in application directory.
Feel free to reach out if you have any questions!
