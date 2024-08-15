import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

class FastApiService {
 Future<Uint8List> sendImageToFastApi(File image) async {
    var request = http.MultipartRequest('POST', Uri.parse('http://192.168.18.50:8000/predict/'));
    request.files.add(await http.MultipartFile.fromPath('file', image.path));
    var response = await request.send();
    if (response.statusCode == 200) {
      var responseBody = await response.stream.toBytes();
      return responseBody;
    } else {
      throw Exception('Failed to load image');
    }
  }
}
