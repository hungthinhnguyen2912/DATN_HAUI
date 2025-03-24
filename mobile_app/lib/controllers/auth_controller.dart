import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';
import 'package:google_sign_in/google_sign_in.dart';

import '../models/User.dart';
import '../views/Widget/bottom_nav_bar.dart';
import '../views/auth/auth_screen.dart';

class AuthController extends GetxController {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final GoogleSignIn _googleSignIn = GoogleSignIn();
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
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
        });
      }
      DocumentSnapshot userDoc =
          await _firestore
              .collection("User")
              .doc(userCredential.user!.uid)
              .get();
      if (userDoc.exists) {
        currentUser.value = UserModel(
          uid: userDoc['uid'],
          name: userDoc['name'],
          email: userDoc['email'],
          createdAt: userDoc['createdAt'],
          avatarUrl:
              userDoc.data().toString().contains('avatarUrl')
                  ? userDoc['avatarUrl']
                  : "",
        );
      }
      Get.off(myBottomNavBar());
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
          uid: userDoc['uid'],
          name: userDoc['name'],
          email: userDoc['email'],
          createdAt: userDoc['createdAt'],
          avatarUrl:
              userDoc.data().toString().contains('avatarUrl')
                  ? userDoc['avatarUrl']
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
        avatarUrl: ""
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
      avatarUrl: "",
    );
    Get.off(AuthScreen());
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


}
