from cProfile import label
from tkinter import Label
from turtle import update
from typing import Text
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
import time
import cv2
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, ObjectProperty
from kivy.clock import Clock
import sys
import numpy as np
from emotion_pytorch import face_mask_prediction
import time
import matplotlib
from kivymd.app import MDApp
from kivy.uix.floatlayout import FloatLayout
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton,MDRectangleFlatButton
from kivy.uix.popup import Popup
from kivy.uix.videoplayer import VideoPlayer
import matplotlib.pyplot as plt
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')

Window.size = (1000, 800)
Builder.load_file('kivyapp_ver2.kv')
glb_label=np.zeros(7)
class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class MenuScreen(Screen):
    
    # global glb_label
    def __init__(self, **kw):
        super().__init__(**kw)
       
        self.dialog=None
        self.image_texture = ObjectProperty(None)
        self.image_capture = ObjectProperty(None)
        self.fn_video = StringProperty()
        self.frame_list=[]


    def video(self):
        global flgPlay
        global click_num
        global times_tracker
        global glb_label
        
        self.fps = 10                 # カメラのFPSを取得
        self.size=(640,480) 
        self.fourcc = cv2.VideoWriter_fourcc('m','p','4','v')            
        flgPlay = not flgPlay
        
        if flgPlay == True:
            self.image_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            Clock.schedule_interval(self.update, 1.0 / 50)
            
        else:
            Clock.unschedule(self.update)
            self.image_capture.release()
            video_writer = cv2.VideoWriter(f'emotion_videos/video{times_tracker}{click_num}.mp4', self.fourcc, self.fps, self.size) 
            for i in range(len(self.frame_list)):
            
                frame=self.frame_list[i]              #変数frameに逆順にしたデータが１フレーム(画像)ずつ入る
                frame=cv2.resize(frame, dsize=self.size)
                video_writer.write(frame)                
            print(len(self.frame_list))
            video_writer.release()
            self.frame_list=[]
            
    def update(self, dt):
            global glb_label
            global click_num
            global times_tracker
            
            ret, frame = self.image_capture.read()

            if ret:
                
                try:
                    
                    frame,labels = face_mask_prediction(frame) # to connect the deeplearning model succesfully with the video capture - application
                    glb_label=glb_label+labels
                    
                    
                    self.frame_list.append(frame)
                    # self.video_writer.write(frame)
                    # width = int(self.self.image_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    # height = int(self.self.image_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # fps = int(self.self.image_capture.get(cv2.CAP_PROP_FPS))
                    # print(width,height,fps)
                    
                    
                    
                    # print(glb_label)
                    
                except:
                    pass
                # カメラ映像をグレースケールに変換
                # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 顔検出
                
            # カメラ映像を上下左右反転
                try:
                    buf = cv2.flip(frame, 0)
                    self.image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                    self.image_texture.blit_buffer(buf.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
                    camera = self.ids['camera']
                    camera.texture = self.image_texture
 
                except:
                    pass
              
                # video.release()       
            # self.video_writer.release()
    def callback(self):
        if not self.dialog:
            self.dialog=MDDialog(
                title="Would you like to see your video?",
                text="Select",
                buttons=[
                    MDFlatButton(
                        text="CANCEL",
                        theme_text_color="Custom",
                        # text_color=(0,0,0,1),
                        on_release=self.close_dialog
                    ),
                    MDRectangleFlatButton(
                        id="neat_button",
                        text="Play the saved video",
                        theme_text_color="Custom",
                        # text_color=(1,0,0,1),
                        on_release=self.show_load
                    ),
                    
                    ]
                
                
            )
        self.dialog.open()
    def close_dialog(self,obj):
        self.dialog.dismiss()
        # self.root.ids.my_label.text="Nah, it's not neat."
    def neat_button(self,obj):
        self.show_load()
        # self.dialog.dismiss()
        # self.root.ids.my_label.text="Yes, It's neat"
    def dismiss_popup(self):
        self._popup.dismiss()

    def load(self, filename):
        try:
            self.fn_video = filename[0]
            # self.dismiss_popup()
            content1 = VideoPlayer(source=self.fn_video ,state='play')
            self._popup = Popup(title="Play video", content=content1,
                                size_hint=(.8, .8))
            self._popup.open()
            # self.show_load()
            
        except Exception as e: print(e)
        
        
        

    def show_load(self,obj):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(.8, .8))
        self._popup.open()
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("images/IMG_{}.png".format(timestr))
        print("Captured")

class Plotting(Screen):
    global glb_label
    def __init__(self, **kw):
        super().__init__(**kw)
       
        self.dialog=None
        self.image_texture = ObjectProperty(None)
        self.image_capture = ObjectProperty(None)
        self.fn_video = StringProperty()
        self.frame_list=[]
    
    def show(self):
        try:
            # fig, ax = plt.subplots()
            plt.clf()
            names =['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            colors = ["Red", "yellow", "purple", "lightpink", "lightcoral","lightblue","gold"]
            
            plt.pie(glb_label,labels=names,counterclock=False, startangle=90,colors=colors,autopct='%.1f%%')
            # box = self.ids.box
            pic=FigureCanvasKivyAgg(plt.gcf())
            try:
                self.ids.box.clear_widgets()
            except Exception as e: print(e)
            
            self.ids.box.add_widget(pic)
            
            # box.add_widget(fig.canvas)
            # glb_label=np.zeros(7)
        # self.ids.test.text=f"{glb_label}"]
        except:
            try:
                self.ids.box.clear_widgets()
            except:
                pass
            button = Button(text="You haven't taken the video")
            self.ids.box.add_widget(button)
    def callback(self):
        if not self.dialog:
            self.dialog=MDDialog(
                title="Would you like to see your video?",
                text="Select",
                buttons=[
                    MDFlatButton(
                        text="CANCEL",
                        theme_text_color="Custom",
                        # text_color=(0,0,0,1),
                        on_release=self.close_dialog
                    ),
                    MDRectangleFlatButton(
                        id="neat_button",
                        text="Play the saved video",
                        theme_text_color="Custom",
                        # text_color=(1,0,0,1),
                        on_release=self.show_load
                    ),
                    
                    ]
                
                
            )
        self.dialog.open()
    def close_dialog(self,obj):
        self.dialog.dismiss()
        # self.root.ids.my_label.text="Nah, it's not neat."
    def neat_button(self,obj):
        self.show_load()
        # self.dialog.dismiss()
        # self.root.ids.my_label.text="Yes, It's neat"
    def dismiss_popup(self):
        self._popup.dismiss()

    def load(self, filename):
        try:
            self.fn_video = filename[0]
            # self.dismiss_popup()
            content1 = VideoPlayer(source=self.fn_video ,state='play')
            self._popup = Popup(title="Play video", content=content1,
                                size_hint=(.8, .8))
            self._popup.open()
            # self.show_load()
            
        except Exception as e: print(e)
        
        
        

    def show_load(self,obj):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(.8, .8))
        self._popup.open()
    def reset_list(self):
        global glb_label
        glb_label=np.zeros(7)  
    def callback_1(self):
        self.ids.test.text="You clicked dots"
    
class MainApp(MDApp):
    def build(self):
        sm=ScreenManager()
        self.theme_cls.theme_style="Light"
        self.theme_cls.primary_pallette="BlueGray"
        # self.theme_cls.accent_pallette="Red"
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(Plotting(name='settings'))
        return sm
      

        
flgPlay=False
times_tracker = time.strftime("%Y%m%d_%H%M%S")
click_num=0
# fps = 10                 # カメラのFPSを取得
# size=(640,480)            # カメラの縦幅を取得
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
# video_writer = cv2.VideoWriter(f'emotion_videos/video{times_tracker}{click_num}.mp4', fourcc, fps, size) 
MainApp().run()
# video_writer.release()