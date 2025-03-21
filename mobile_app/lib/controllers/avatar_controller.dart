import 'dart:convert';
import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class AvatarController extends GetxController {
  final Rx<File?> avatar = Rx<File?>(null);
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final RxString avatarUrl = RxString("");

  @override
  void onInit() {
    super.onInit();
    fetchLinkAvatar();
  }

  Future<void> fetchLinkAvatar() async {
    DocumentSnapshot userDoc =
        await _firestore.collection("User").doc(_auth.currentUser!.uid).get();
    if (userDoc.exists && userDoc.data() != null) {
      avatarUrl.value = userDoc['avatarUrl'];
    } else {
      Get.snackbar("Error", "");
    }
  }

  Future<void> galleryAvatar() async {
    final pickedFile = await ImagePicker().pickImage(
      source: ImageSource.gallery,
    );
    if (pickedFile == null) {
      Get.snackbar("Error", "Không có ảnh được chọn.");
      return;
    }
    avatar.value = File(pickedFile.path);
  }

  Future<void> cameraAvatar() async {
    final pickedFile = await ImagePicker().pickImage(
      source: ImageSource.camera,
    );
    if (pickedFile == null) {
      Get.snackbar("Error", "No image selected.");
      return;
    }
    avatar.value = File(pickedFile.path);
  }

  Future<void> postAvatarToCloudinary() async {
    if (avatar.value == null) {
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
        await http.MultipartFile.fromPath("file", avatar.value!.path),
      );

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      var jsonResponse = json.decode(responseData);

      if (response.statusCode == 200) {
        avatarUrl.value = jsonResponse["secure_url"];
        print("Uploaded Avatar URL: ${avatarUrl.value}");
        Get.snackbar("Success", "Image uploaded successfully!");
      } else {
        Get.snackbar("Error", "Failed to upload image");
      }
    } catch (e) {
      Get.snackbar("Error", "An error occurred while uploading");
      print("Upload error: $e");
    }
  }

  // Cloudinary does not support for put method
  // Delete old avatar first then post new avatar
  Future<void> putAvatarToCloudinary() async {}
}
