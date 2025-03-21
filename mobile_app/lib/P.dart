import 'package:get/get.dart';
import 'controllers/avatar_controller.dart';
import 'controllers/auth_controller.dart';

class P {
  static void initialController () {
    Get.put(AuthController());
    // Get.put(AvatarController());
  }
  static AuthController get auth  => Get.find();
  static AvatarController get avatar => Get.find();
}