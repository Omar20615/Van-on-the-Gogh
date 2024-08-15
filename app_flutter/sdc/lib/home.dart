import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  final ImagePicker _picker = ImagePicker();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Van on the Gogh'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image != null
                ? Container(
                    width: double.infinity,
                    height: 300, // You can adjust the height as per your requirement
                    child: Image.file(
                      _image!,
                      fit: BoxFit.cover, // This will scale the image to fit the container
                    ),
                  )
                : Text("No image selected"),

            SizedBox(height: 20),

            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton(
                  onPressed: () async {
                    final XFile? pickedFile = await _picker.pickImage(source: ImageSource.gallery);
                    setState(() {
                      if (pickedFile != null) {
                        _image = File(pickedFile.path);
                      } else {
                        _image = null;
                      }
                    });
                  },
                  child: Text('Browse'),
                ),
                ElevatedButton(
                  onPressed: () async {
                    final XFile? pickedFile = await _picker.pickImage(source: ImageSource.camera);
                    setState(() {
                      if (pickedFile != null) {
                        _image = File(pickedFile.path);
                      } else {
                        _image = null;
                      }
                    });
                  },
                  child: Text('Capture'),
                ),
              ],
            ),

            SizedBox(height: 20),

            ElevatedButton(
              onPressed: _image != null
                  ? () {
                      Navigator.pushNamed(context, '/upload', arguments: _image);
                    }
                  : null,
              child: Text('Send Image to Fast API'),
            ),
          ],
        ),
      ),
    );
  }
}
