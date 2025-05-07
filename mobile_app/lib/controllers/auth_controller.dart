import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:mobile_app/views/setting/items_setting_page/confirm_page.dart';

import '../models/User.dart';
import '../views/Widget/bottom_nav_bar.dart';
import '../views/auth/auth_screen.dart';

class AuthController extends GetxController {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final GoogleSignIn _googleSignIn = GoogleSignIn();
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  RxBool isGoogleUser = false.obs;
  Rxn<User> firebaseUser = Rxn<User>();
  Rx<UserModel?> currentUser = Rx<UserModel?>(null);

  Future<void> loginWithGoogle() async {
    try {
      final GoogleSignInAccount? googleUser = await _googleSignIn.signIn();
      if (googleUser == null) return;

      final GoogleSignInAuthentication googleAuth =
          await googleUser.authentication;
      final AuthCredential credential = GoogleAuthProvider.credential(
        idToken: googleAuth.idToken,
        accessToken: googleAuth.accessToken,
      );

      UserCredential userCredential = await _auth.signInWithCredential(
        credential,
      );
      bool isNewUser = userCredential.additionalUserInfo!.isNewUser;

      if (isNewUser) {
        await _firestore.collection("User").doc(userCredential.user!.uid).set({
          "name": googleUser.displayName,
          "email": googleUser.email,
          "createdAt": DateTime.now(),
          "uid": userCredential.user!.uid,
          "avatarUrl": "",
          "publicIdAvatar": googleUser.photoUrl,
        });
      }
      DocumentSnapshot userDoc =
          await _firestore
              .collection("User")
              .doc(userCredential.user!.uid)
              .get();
      if (userDoc.exists) {
        currentUser.value = UserModel(
          phone: userDoc['phone'],
          uid: userDoc['uid'],
          name: userDoc['name'],
          email: userDoc['email'],
          createdAt: userDoc['createdAt'],
          avatarUrl:
              userDoc.data().toString().contains('avatarUrl')
                  ? userDoc['avatarUrl']
                  : "",
          publicIdAvatar:
              userDoc.data().toString().contains('publicId')
                  ? userDoc['publicIdAvatar']
                  : "",
        );
      }
      Get.off(myBottomNavBar());
      isGoogleUser.value = true;
    } on FirebaseAuthException catch (e) {
      if (e.code == 'account-exists-with-different-credential') {
        Get.snackbar(
          "Error",
          "Tài khoản đã được đăng ký bằng phương thức khác.",
        );
      } else {
        Get.snackbar("Error", "Có lỗi xảy ra: ${e.message}");
      }
    }
  }

  Future<void> login(String email, String password) async {
    try {
      await _auth.signInWithEmailAndPassword(email: email, password: password);
      DocumentSnapshot userDoc =
          await _firestore.collection("User").doc(_auth.currentUser!.uid).get();
      if (userDoc.exists && userDoc.data() != null) {
        currentUser.value = UserModel(
          phone: userDoc['phone'],
          uid: userDoc['uid'],
          name: userDoc['name'],
          email: userDoc['email'],
          createdAt: userDoc['createdAt'],
          avatarUrl:
              userDoc.data().toString().contains('avatarUrl')
                  ? userDoc['avatarUrl']
                  : "",
          publicIdAvatar:
              userDoc.data().toString().contains('publicId')
                  ? userDoc['publicIdAvatar']
                  : "",
        );
        Get.off(myBottomNavBar());
      } else {
        Get.snackbar("Error", "Tài khoản không tồn tại.");
      }
    } on FirebaseAuthException {
      Get.snackbar("Error", "Something went wrong");
    }
  }

  Future<void> register(String email, String password, String name) async {
    try {
      UserCredential userCredential = await _auth
          .createUserWithEmailAndPassword(email: email, password: password);
      UserModel user = UserModel(
        name: name,
        email: email,
        createdAt: Timestamp.now(),
        uid: userCredential.user!.uid,
        avatarUrl: "",
        publicIdAvatar: "",
        phone: '',
      );
      currentUser.value = user;
      await _firestore
          .collection("User")
          .doc(userCredential.user!.uid)
          .set(user.toJson());
      Get.defaultDialog(
        title: "Complete",
        onConfirm: () => Get.off(myBottomNavBar()),
      );
    } on FirebaseAuthException catch (e) {
      if (e.code == 'email-already-in-use') {
        Get.snackbar("Error", "Email này đã được sử dụng.");
      } else {
        Get.snackbar("Error", "Đăng ký thất bại: ${e.message}");
      }
    }
  }

  void logOut() async {
    await _auth.signOut();
    firebaseUser.value = null;
    currentUser.value = UserModel(
      uid: "",
      name: "",
      email: "",
      createdAt: Timestamp(0, 0),
      avatarUrl: "", phone: '',
    );
    Get.offAll(() => AuthScreen());
    isGoogleUser.value = false;
  }

  Future<void> resetPassword(String email) async {
    try {
      await _auth.sendPasswordResetEmail(email: email);
      Get.snackbar("Success", "Vui lòng kiểm tra email của bạn.");
    } on FirebaseAuthException catch (e) {
      print(e.message);
      Get.snackbar("Error", "Có lỗi xảy ra: ${e.message}");
    }
  }

  Future<void> authenticationPassword(String email, String currentPass) async {
    Get.dialog(
      Center(child: CircularProgressIndicator()),
      barrierDismissible: false,
    );
    try {
      AuthCredential authCredential = EmailAuthProvider.credential(
        email: email,
        password: currentPass,
      );
      if (firebaseUser.value == null) {
        Get.snackbar("Lỗi", "Người dùng chưa đăng nhập");
        return;
      }
      await firebaseUser.value?.reauthenticateWithCredential(authCredential);
      Get.to(ConfirmPage());
    } on FirebaseAuthException catch (e) {
      Get.snackbar("Error", "Có lỗi xảy ra: ${e.message}");
    } finally {
      Get.back();
    }
  }

  Future<void> updatePassword(String newPass) async {
    try {
      await firebaseUser.value?.updatePassword(newPass);
      Get.snackbar("Success", "Cập nhật mật khẩu thành công.");
    } on FirebaseAuthException catch (e) {
      Get.snackbar("Error", "Có lỗi xảy ra: ${e.message}");
      Get.off(AuthScreen());
    }
  }

  Future<void> editProfile(String email, String name) async {
    try {
      await FirebaseFirestore.instance
          .collection("User")
          .doc(currentUser.value!.uid)
          .update({"name": name, "email": email});
    } catch (e) {
      print("Edit failed: ${e.toString()}");
    }
  }
}
