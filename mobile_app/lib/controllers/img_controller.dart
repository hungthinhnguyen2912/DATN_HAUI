import 'dart:convert';
import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:cloudinary/cloudinary.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

import '../P.dart';

class ImageController extends GetxController {
  late Rx<File?> image = Rx<File?>(null);
  final RxString imageUrl = RxString("");
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  RxString publicId = "".obs;

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

  Future<void> postImgCloudinary() async {
    if (image.value == null) {
      print("No image selected");
      Get.snackbar("Error", "No image selected");
      return;
    }
    try {
      print('Start upload image to Cloudinary...');
      var response = await P.cloudinary.upload(
        file: File(image.value!.path).path,
        resourceType: CloudinaryResourceType.image,
        folder: "Vegetables",
        progressCallback: (count, total) {
          print("Upload image success with progress: $count/$total");
        },
      );
      if (response.isSuccessful) {
        imageUrl.value = response.secureUrl ?? "";
        publicId.value = response.publicId ?? "";
        print("Uploaded Image URL: ${imageUrl.value}");
        Get.snackbar("Success", "Image upload complete");
      } else {
        print("Error upload image");
      }
    } catch (e) {
      Get.snackbar("Error", "An error occurred while uploading");
      print("Lỗi upload: $e");
    }
  }

}
