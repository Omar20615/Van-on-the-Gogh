import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:sdc/api_call.dart';

class UploadPage extends StatefulWidget {
  @override
  _UploadPageState createState() => _UploadPageState();
}

class _UploadPageState extends State<UploadPage> {
  Uint8List? _returnedImage;
  File? _image;

  FastApiService _fastApiService = FastApiService();

  bool _isLoading = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _image = ModalRoute.of(context)?.settings.arguments as File?;
    if (_image!= null) {
      _sendImageToFastApi();
    }
  }

  void _sendImageToFastApi() async {
    setState(() {
      _isLoading = true;
    });
    final returnedImage = await _fastApiService.sendImageToFastApi(_image!);
    setState(() {
      _returnedImage = returnedImage;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Upload Page'),
      ),
      body: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              _image!= null
               ? Image.file(
                    _image!,
                    fit: BoxFit.contain, // This will scale the image to fit the container
                  )
                : Text("No image selected"),
        
              SizedBox(height: 20),
        
              Text("Before"),
        
              SizedBox(height: 100),
        
              _isLoading
               ? Center(
                      child: CircularProgressIndicator(),
                    )
                : _returnedImage!= null
                 ? Image.memory(
                      _returnedImage!,
                      fit: BoxFit.contain, // This will scale the image to fit the container
                    )
                  : Text("In progress"),
        
              SizedBox(height: 20),
        
              Text("After"),
            ],
          ),
        ),
      ),
    );
  }
}