import 'package:get/get.dart';
import 'package:vegetable_classification/controllers/auth_controller.dart';

class P {
  static void initialController () {
    Get.put(AuthController());
}
  static AuthController get auth  => Get.find();
}