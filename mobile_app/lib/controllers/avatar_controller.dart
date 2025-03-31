import 'dart:convert';
import 'dart:io';
import 'package:crypto/crypto.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:cloudinary/cloudinary.dart';
import '../P.dart';

class AvatarController extends GetxController {
  final Rx<File?> avatar = Rx<File?>(null);
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final RxString avatarUrl = RxString("");
  final RxString publicId = RxString("");


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
      publicId.value = userDoc['publicIdAvatar'];
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

  Future<void> postCloudinary() async {
    if (avatar.value == null) {
      print("No image selected");
      Get.snackbar("Error", "No image selected");
      return;
    }
    try {
      var response = await P.cloudinary.upload(
        file: File(avatar.value!.path).path,
        resourceType: CloudinaryResourceType.image,
        folder: "Avatars",
        fileName: "user_avatars_${_auth.currentUser!.uid}",
        progressCallback: (count, total) {
          print("Upload avatar success with progress: $count/$total");
        },
      );
      if (response.isSuccessful) {
        avatarUrl.value = response.secureUrl ?? "";
        publicId.value = response.publicId ?? "";
        print("Uploaded Avatar URL: ${avatarUrl.value}");
        await _firestore.collection("User").doc(_auth.currentUser!.uid).update({
          "avatarUrl": avatarUrl.value,
          "publicIdAvatar": publicId.value,
        });
        Get.snackbar("Success", "Avatar upload complete");
      } else {
        print("Error upload image");
      }
    } catch (e) {
      print("Lỗi post Cloudinary: ${e.toString()}");
    }
  }

  Future<void> DeleteAndPostCloudinary() async {
    try {
      P.cloudinary.destroy(
        publicId.value,
        resourceType: CloudinaryResourceType.image,
        invalidate: false,
      );
    } catch (e) {
      print("Lỗi delete: ${e.toString()}");
    }
    postCloudinary();
  }
}
