import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';
import 'package:mobile_app/views/auth/auth_phone/verify_code.dart';

import '../string_extension.dart';

class AuthPhoneController extends GetxController {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  RxString verificationID = "".obs;



  Future<void> sendOTP(String phoneNumber) async {
    try {
      await _auth.verifyPhoneNumber(
        phoneNumber: formatPhoneNumber(phoneNumber),
        verificationCompleted: (PhoneAuthCredential credential) {},
        codeSent: (String verificationId, int? resendToken) {
          verificationID.value = verificationId;
            print("OTP sent to $phoneNumber");
        },
        codeAutoRetrievalTimeout: (String verificationId) {
          print("Timeout: $verificationId");
          verificationID.value = verificationId;
        },
        timeout: const Duration(seconds: 60),
        verificationFailed: (FirebaseAuthException error) {
          print(error.toString());
        },
      );
      Get.to(() => VerifyCode());
    } on FirebaseAuthException catch (e) {
      print("Firebase auth exception");
      print(e.message);
      Get.snackbar("Error", "Có lỗi xảy ra: ${e.message}");
    }
  }

  Future<void> verifyOTP(String smsCode) async {
    try {
      final credential = PhoneAuthProvider.credential(
        verificationId: verificationID.value,
        smsCode: smsCode,
      );
      await _auth.signInWithCredential(credential);
      await FirebaseFirestore.instance
          .collection("User")
          .doc(FirebaseAuth.instance.currentUser!.uid)
          .set({
            "phone": formatPhoneNumber(
              FirebaseAuth.instance.currentUser!.phoneNumber!,
            ),
            "uid": FirebaseAuth.instance.currentUser!.uid,
            "name": "user${generateRandomString(10)}",
            "email": "",
            "createdAt": DateTime.now(),
            "avatarUrl": "",
            "publicIdAvatar": "",
          });
      print("Xác thực thành công!");
      Get.offNamed("/home");
    } catch (e) {
      print("Xác thực OTP thất bại: $e");
    }
  }
}
