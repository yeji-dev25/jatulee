package com.p_project.book;

import com.p_project.diary.DiaryDTO;
import com.p_project.oauth2.CustomOAuth2User;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/bookreport")
@CrossOrigin(origins = "*") // React 앱과 통신 허용
public class BookController {

    private final BookService bookService;

    // 전체 독후감 리스트 조회
    @GetMapping("/list")
    public ResponseEntity<List<DiaryDTO>> getAllReports(Authentication auth) {
        CustomOAuth2User principal = (CustomOAuth2User) auth.getPrincipal();
        List<DiaryDTO> reports = bookService.getAllReports(principal.getUserId());
        return ResponseEntity.ok(reports);
    }

    @PostMapping("/{id}/complete")
    public ResponseEntity<?> complete(@PathVariable Long id) {
        return ResponseEntity.ok(bookService.complete(id));
    }

    @GetMapping("/me/books")
    public ResponseEntity<List<BookDTO>> getMyBookSessionList() {

        List<BookDTO> reviewedBooks = bookService.getMyBookSessions();

        return ResponseEntity.ok(reviewedBooks);
    }
}
