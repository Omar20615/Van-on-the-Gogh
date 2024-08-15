import 'package:flutter/material.dart';
import 'package:sdc/result_page.dart';
import 'home.dart';
void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
 Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Van on the Gogh',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => MyHomePage(),
        '/upload': (context) => UploadPage(),
      },
    );
  }
}
