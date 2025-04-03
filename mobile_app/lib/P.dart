import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:cloudinary/cloudinary.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';
import 'package:mobile_app/controllers/auth_phone_controller.dart';
import 'package:mobile_app/controllers/classification_controller.dart';
import 'package:mobile_app/controllers/history_controller.dart';
import 'package:mobile_app/controllers/img_controller.dart';
import 'controllers/avatar_controller.dart';
import 'controllers/auth_controller.dart';

class P {
  static void initialController () {
    Get.put(AuthController());
    // Get.put(AvatarController());
    Get.put(AuthPhoneController());
    Get.put(ImageController());
    Get.put(ClassificationController());
    Get.put(HistoryController());
  }
  static final cloudinary = Cloudinary.signedConfig(
    apiKey: "328643715299492",
    apiSecret: "PPjRIC_NRfOkxb5kXqkmvKSjbeg",
    cloudName: "graduation",
  );
  static FirebaseAuth get authFirebase => FirebaseAuth.instance;
  static FirebaseFirestore get fireStore => FirebaseFirestore.instance;
  static AuthController get auth  => Get.find();
  static AvatarController get avatar => Get.find();
  static AuthPhoneController get phone => Get.find();
  static ImageController get image => Get.find();
  static ClassificationController get classification => Get.find();
  static HistoryController get history => Get.find();

}