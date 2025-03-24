import 'dart:convert';
import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class ImageController extends GetxController {
  final Rx<File?> image = Rx<File?>(null);
  final RxString imageUrl = RxString("");
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;

  Future<void> galleryImage() async {
    final pickedFile = await ImagePicker().pickImage(
      source: ImageSource.gallery,
    );
    if (pickedFile == null) {
      Get.snackbar("Error", "Không có ảnh được chọn.");
      return;
    }
    image.value = File(pickedFile.path);
  }

  Future<void> cameraImage() async {
    final pickedFile = await ImagePicker().pickImage(
      source: ImageSource.camera,
    );
    if (pickedFile == null) {
      Get.snackbar("Error", "No image selected.");
      return;
    }
    image.value = File(pickedFile.path);
  }

  Future<void> uploadToCloudinary() async {
    if (image.value == null) {
      print("No image selected");
      Get.snackbar("Error", "No image selected");
      return;
    }

    try {
      print('Start upload image to Cloudinary...');
      String cloudName = "dcqn3q7tg";
      String uploadPreset = "Vegetable";

      Uri url = Uri.parse(
        "https://api.cloudinary.com/v1_1/$cloudName/image/upload",
      );

      var request = http.MultipartRequest("POST", url);
      request.fields["upload_preset"] = uploadPreset;
      request.fields["folder"] = "Vegetables";
      request.files.add(
        await http.MultipartFile.fromPath("file", image.value!.path),
      );

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      var jsonResponse = json.decode(responseData);

      if (response.statusCode == 200) {
        imageUrl.value = jsonResponse["secure_url"];
        print("Uploaded Avatar URL: ${imageUrl.value}");
        Get.snackbar("Success", "Image uploaded successfully!");
      } else {
        Get.snackbar("Error", "Failed to upload image");
      }
    } catch (e) {
      Get.snackbar("Error", "An error occurred while uploading");
      print("Upload error: $e");
    }
  }
}
