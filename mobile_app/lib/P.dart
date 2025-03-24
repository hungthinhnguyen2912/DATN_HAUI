import 'package:get/get.dart';
import 'package:mobile_app/controllers/auth_phone_controller.dart';
import 'package:mobile_app/controllers/img_controller.dart';
import 'controllers/avatar_controller.dart';
import 'controllers/auth_controller.dart';

class P {
  static void initialController () {
    Get.put(AuthController());
    // Get.put(AvatarController());
    Get.put(AuthPhoneController());
    Get.put(ImageController());
  }
  static AuthController get auth  => Get.find();
  static AvatarController get avatar => Get.find();
  static AuthPhoneController get phone => Get.find();
  static ImageController get image => Get.find();
}