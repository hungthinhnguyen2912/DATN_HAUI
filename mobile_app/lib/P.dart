import 'package:get/get.dart';

import 'controllers/auth_controller.dart';

class P {
  static void initialController () {
    Get.put(AuthController());
  }
  static AuthController get auth  => Get.find();
}