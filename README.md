# Facee

Abstract
Facee is a web app that helps visually impaired persons with to recognize people around them.
The project was made in collaboration with a PHD student in Cognitive Science that specializes in
human disabilities, especially visual impairment

Technolegy
With the help of a simple computer camera, the app learns the user’s friends faces, and saves them in a DB
The app recognizes faces using a convolutional neural networks algorithm, and compares them to the DB
If the face is match, the app will announce vocally the name of the person

Libraries
Google API open face - open code for detecting whether two given faces match
MeSpeak - open code for text to speech voice reader
