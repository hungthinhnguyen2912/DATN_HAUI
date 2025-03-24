import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';

class AuthPhoneController extends GetxController {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  String formatPhoneNumber(String phoneNumber) {
    if (phoneNumber.startsWith("0")) {
      return "+84${phoneNumber.substring(1)}";
    } else {
      return phoneNumber;
    }
  }

  Future<void> sendOTP(String phoneNumber) async {
    try {
      await _auth.verifyPhoneNumber(
        phoneNumber: formatPhoneNumber(phoneNumber),
        verificationCompleted: (PhoneAuthCredential credential) {},
        codeSent: (String verificationId, int? resendToken) {
          print("OTP sent to $phoneNumber");
        },
        codeAutoRetrievalTimeout: (String verificationId) {
          print("Timeout: $verificationId");
        },
        timeout: const Duration(seconds: 60),
        verificationFailed: (FirebaseAuthException error) {
          print(error.toString());
        },
      );
    } on FirebaseAuthException catch (e) {
      print("Firebase auth exception");
      print(e.message);
      Get.snackbar("Error", "Có lỗi xảy ra: ${e.message}");
    }
  }
}
